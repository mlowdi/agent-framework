import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { once } from 'node:events';
import { MessageChannel, type MessagePort } from 'node:worker_threads';
import { JsStore } from '@animalabs/chronicle';
import { ContextManager } from '@animalabs/context-manager';
import { AgentFramework } from '../src/index.js';
import type { Module, ModuleContext, ToolCall, ToolResult } from '../src/types/index.js';
import { MockMembrane } from './helpers/mock-membrane.js';

class StartupResource implements Module {
  private channel?: MessageChannel;
  constructor(readonly name: string, private resources: Set<MessagePort>,
    private startError?: Error, private stopError?: Error) {}
  async start(context: ModuleContext): Promise<void> {
    context.setState({ initialized: true });
    this.channel = new MessageChannel();
    for (const port of [this.channel.port1, this.channel.port2]) { port.ref(); this.resources.add(port); }
    if (this.startError) throw this.startError;
  }
  async stop(): Promise<void> {
    if (this.channel) {
      const ports = [this.channel.port1, this.channel.port2];
      const closed = Promise.all(ports.map(port => once(port, 'close')));
      for (const port of ports) port.close();
      await closed;
      for (const port of ports) this.resources.delete(port);
    }
    if (this.stopError) throw this.stopError;
  }
  getTools() { return []; }
  async onProcess() { return {}; }
  async handleToolCall(_call: ToolCall): Promise<ToolResult> { throw new Error('no tools'); }
}

test('failed creation releases agents retained by module context and leaves an app-owned store reusable', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'af-create-agent-cleanup-'));
  const store = JsStore.openOrCreate({ path: join(dir, 'store') });
  const resources = new Set<MessagePort>();
  let captured: ModuleContext | undefined;
  class CaptureContext extends StartupResource {
    override async start(context: ModuleContext): Promise<void> {
      captured = context;
      assert.equal(context.getAgents().length, 1, 'an agent exists before the later startup failure');
      await super.start(context);
    }
  }
  const failure = new Error('later module startup failed');
  const baseline = process.listeners('SIGUSR2');
  try {
    await assert.rejects(AgentFramework.create({ store, operatorLogPath: false,
      membrane: new MockMembrane().asMembrane(),
      agents: [{ name: 'assistant', model: 'fixture', systemPrompt: 'test' }],
      modules: [new CaptureContext('capture', resources), new StartupResource('failure', resources, failure)],
      syncIntervalMs: 0,
    }), error => error === failure);
    assert.deepEqual(process.listeners('SIGUSR2'), baseline);
    assert.ok(captured);
    assert.deepEqual(captured.getAgents(), [], 'a retained adapter context cannot keep failed framework agents live');
    assert.equal(resources.size, 0);
    const replacement = await AgentFramework.create({ store, operatorLogPath: false,
      membrane: new MockMembrane().asMembrane(),
      agents: [{ name: 'assistant', model: 'fixture', systemPrompt: 'replacement' }], modules: [], syncIntervalMs: 0 });
    await replacement.stop();
    assert.ok(store.currentSequence() > 0, 'neither failed nor replacement framework owns this store');
  } finally {
    for (const port of resources) port.close();
    store.close(); rmSync(dir, { recursive: true, force: true });
  }
});

for (const invalid of [
  { state: { agentRuntimeSettings: { assistant: { sameRoundThinkTextPolicy: 'invalid' } } }, error: /Invalid persisted sameRoundThinkTextPolicy/ },
  { state: { toolResultGuards: { assistant: 'invalid' } }, error: /Invalid persisted tool_result_guard/ },
]) {
  test(`pre-registration startup failure releases context listeners (${invalid.error.source})`, async t => {
    const dir = mkdtempSync(join(tmpdir(), 'af-create-unregistered-agent-'));
    const store = JsStore.openOrCreate({ path: join(dir, 'store') });
    store.registerState({ id: 'framework/state', strategy: 'snapshot' });
    store.setStateJson('framework/state', invalid.state);
    const opened: ContextManager[] = [];
    let notifications = 0;
    const realOpen = ContextManager.open.bind(ContextManager);
    // Capture a real returned resource to exercise its behavior after failure;
    // no fake manager, private-state patch, or close-call-count assertion.
    t.mock.method(ContextManager, 'open', async (config: Parameters<typeof ContextManager.open>[0]) => {
      const cm = await realOpen(config);
      opened.push(cm);
      cm.addMessage('User', [{ type: 'text', text: 'initialized context is observable' }]);
      return cm;
    });
    try {
      await assert.rejects(AgentFramework.create({ store, operatorLogPath: false,
        membrane: new MockMembrane().asMembrane(),
        agents: [{ name: 'assistant', model: 'fixture', systemPrompt: 'test', strategy: {
          name: 'owned-listener', checkReadiness: () => ({ ready: true }), select: () => [],
          onNewMessage: async () => { notifications++; },
        } }], modules: [], syncIntervalMs: 0,
      }), invalid.error);
      assert.equal(notifications, 1, 'the real initialized context has an active strategy before the failure');
      assert.equal(opened.length, 1);
      const sequence = store.currentSequence();
      opened[0].addMessage('User', [{ type: 'text', text: 'retained handle after failed startup' }]);
      assert.ok(store.currentSequence() > sequence, 'the caller-owned store is still usable');
      assert.equal(notifications, 1, 'a failed unregistered context cannot continue notifying its former strategy');
      t.mock.restoreAll();
      store.setStateJson('framework/state', {});
      const replacement = await AgentFramework.create({ store, operatorLogPath: false,
        membrane: new MockMembrane().asMembrane(),
        agents: [{ name: 'assistant', model: 'fixture', systemPrompt: 'replacement' }], modules: [], syncIntervalMs: 0 });
      await replacement.stop();
      assert.ok(store.currentSequence() > sequence);
    } finally {
      for (const cm of opened) cm.close();
      store.close(); rmSync(dir, { recursive: true, force: true });
    }
  });
}

test('diagnostic SIGUSR2 ownership survives 12 sequential cycles and simultaneous frameworks', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'af-signal-lifecycle-'));
  const store = JsStore.openOrCreate({ path: join(dir, 'store') });
  const entry = process.listeners('SIGUSR2');
  let sentinelCalls = 0;
  const sentinel = () => { sentinelCalls++; };
  process.on('SIGUSR2', sentinel);
  const baseline = process.listeners('SIGUSR2');
  const warnings: Error[] = [];
  const captureWarning = (warning: Error) => { warnings.push(warning); };
  process.on('warning', captureWarning);
  const live = new Set<AgentFramework>();
  const create = async (name: string) => {
    const framework = await AgentFramework.create({ store, operatorLogPath: false,
      membrane: new MockMembrane().asMembrane(),
      agents: [{ name, model: 'fixture', systemPrompt: 'test' }], modules: [], syncIntervalMs: 0 });
    live.add(framework);
    return framework;
  };
  try {
    for (let cycle = 0; cycle < 12; cycle++) {
      const framework = await create(`cycle-${cycle}`);
      assert.equal(process.listenerCount('SIGUSR2'), baseline.length + 1);
      await framework.stop();
      live.delete(framework);
      assert.deepEqual(process.listeners('SIGUSR2'), baseline, `cycle ${cycle} restores exact entry listeners`);
    }
    await new Promise<void>(resolve => setImmediate(resolve));
    assert.equal(warnings.filter(warning => warning.name === 'MaxListenersExceededWarning').length, 0);

    const first = await create('first-live');
    const second = await create('second-live');
    const handlers = process.listeners('SIGUSR2').filter(handler => !baseline.includes(handler));
    assert.equal(handlers.length, 2);
    const [firstHandler, secondHandler] = handlers;
    assert.notEqual(firstHandler, secondHandler);
    const output: string[] = [];
    const originalError = console.error;
    console.error = (...args: unknown[]) => { output.push(args.map(String).join(' ')); };
    try { process.emit('SIGUSR2'); } finally { console.error = originalError; }
    assert.equal(sentinelCalls, 1);
    const diagnostics = output.filter(line => line.startsWith('[diagnostics] '))
      .map(line => JSON.parse(line.slice('[diagnostics] '.length)));
    assert.deepEqual(diagnostics.map(dump => dump.agents), [['first-live'], ['second-live']]);
    for (const dump of diagnostics) {
      assert.equal(typeof dump.at, 'string');
      assert.equal(dump.gate, null);
      assert.deepEqual(dump.activeStreams, []);
      assert.equal(dump.pendingRequests, 0);
    }
    await first.stop();
    live.delete(first);
    assert.deepEqual(process.listeners('SIGUSR2'), [...baseline, secondHandler]);
    // Listener release is idempotent; this does not promise a new public restart lifecycle.
    await first.stop();
    assert.deepEqual(process.listeners('SIGUSR2'), [...baseline, secondHandler]);
    await second.stop();
    live.delete(second);
    assert.deepEqual(process.listeners('SIGUSR2'), baseline);
  } finally {
    for (const framework of live) await framework.stop();
    process.removeListener('warning', captureWarning);
    process.removeListener('SIGUSR2', sentinel);
    assert.deepEqual(process.listeners('SIGUSR2'), entry);
    store.close(); rmSync(dir, { recursive: true, force: true });
  }
});

test('stop releases its diagnostic handler before asynchronous module shutdown fails', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'af-signal-stop-failure-'));
  const store = JsStore.openOrCreate({ path: join(dir, 'store') });
  const baseline = process.listeners('SIGUSR2');
  const resources = new Set<MessagePort>();
  const failure = new Error('module shutdown failed');
  const framework = await AgentFramework.create({ store, operatorLogPath: false,
    membrane: new MockMembrane().asMembrane(),
    agents: [{ name: 'assistant', model: 'fixture', systemPrompt: 'test' }],
    modules: [new StartupResource('shutdown-failure', resources, undefined, failure)], syncIntervalMs: 0 });
  try {
    assert.equal(process.listenerCount('SIGUSR2'), baseline.length + 1);
    await assert.rejects(framework.stop(), error => error === failure ||
      (error instanceof AggregateError && error.errors.includes(failure)));
    assert.deepEqual(process.listeners('SIGUSR2'), baseline);
    assert.equal(resources.size, 0);
    await framework.stop();
    assert.deepEqual(process.listeners('SIGUSR2'), baseline);
    assert.ok(store.currentSequence() > 0, 'caller-owned store remains usable despite stop failure');
  } finally {
    for (const port of resources) port.close();
    store.close(); rmSync(dir, { recursive: true, force: true });
  }
});

test('unsupported SIGUSR2 registration remains nonfatal and leaves unrelated listeners intact', async t => {
  const dir = mkdtempSync(join(tmpdir(), 'af-signal-unsupported-'));
  const store = JsStore.openOrCreate({ path: join(dir, 'store') });
  const baseline = process.listeners('SIGUSR2');
  const originalOn = process.on;
  t.mock.method(process, 'on', function (this: NodeJS.Process, event: string | symbol, listener: (...args: unknown[]) => void) {
    if (event === 'SIGUSR2') throw new Error('signal unsupported');
    return originalOn.call(this, event, listener);
  });
  try {
    const framework = await AgentFramework.create({ store, operatorLogPath: false,
      membrane: new MockMembrane().asMembrane(),
      agents: [{ name: 'assistant', model: 'fixture', systemPrompt: 'test' }], modules: [], syncIntervalMs: 0 });
    assert.deepEqual(process.listeners('SIGUSR2'), baseline);
    await framework.stop();
    assert.deepEqual(process.listeners('SIGUSR2'), baseline);
  } finally {
    t.mock.restoreAll();
    store.close(); rmSync(dir, { recursive: true, force: true });
  }
});

for (const callerOwned of [false, true]) {
  for (const cleanupFails of [false, true]) {
    test(`failed create cleans partial modules before store reopen (caller owned=${callerOwned}, cleanup fails=${cleanupFails})`, async () => {
      const dir = mkdtempSync(join(tmpdir(), 'af-create-cleanup-'));
      const storePath = join(dir, 'store');
      const store = callerOwned ? JsStore.openOrCreate({ path: storePath }) : undefined;
      const resources = new Set<MessagePort>();
      const startError = new Error('second module initialization failed');
      const stopError = cleanupFails ? new Error('first module cleanup failed') : undefined;
      const baseline = process.listeners('SIGUSR2');
      try {
        const create = AgentFramework.create({
          ...(store ? { store } : { storePath }), operatorLogPath: false,
          membrane: new MockMembrane().asMembrane(),
          agents: [{ name: 'assistant', model: 'fixture', systemPrompt: 'test' }],
          modules: [new StartupResource('first', resources, undefined, stopError),
            new StartupResource('second', resources, startError)], syncIntervalMs: 0,
        });
        await assert.rejects(create, error => {
          if (!cleanupFails) return error === startError;
          assert.ok(error instanceof AggregateError);
          assert.equal(error.cause, startError);
          assert.equal(error.errors[0], startError);
          const causes = (value: unknown): unknown[] => value instanceof AggregateError ? value.errors.flatMap(causes) : [value];
          assert.deepEqual(causes(error), [startError, stopError], 'original and cleanup failures both survive');
          return true;
        });
        assert.deepEqual(process.listeners('SIGUSR2'), baseline);
        assert.equal(resources.size, 0, 'both completed and partially started modules relinquished live resources');
        // A caller-owned store stays usable; a factory-owned store can reopen.
        const reopened = store ?? JsStore.openOrCreate({ path: storePath });
        try {
          assert.ok(reopened.currentSequence() > 0);
          const good = await AgentFramework.create({ store: reopened, operatorLogPath: false,
            membrane: new MockMembrane().asMembrane(),
            agents: [{ name: 'assistant', model: 'fixture', systemPrompt: 'test' }], modules: [], syncIntervalMs: 0 });
          await good.stop();
          assert.ok(reopened.currentSequence() > 0, 'framework never closes the caller-owned store');
        } finally { if (!store) reopened.close(); }
      } finally {
        for (const port of resources) port.close();
        store?.close();
        rmSync(dir, { recursive: true, force: true });
      }
    });
  }
}
