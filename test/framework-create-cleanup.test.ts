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
  try {
    await assert.rejects(AgentFramework.create({ store, operatorLogPath: false,
      membrane: new MockMembrane().asMembrane(),
      agents: [{ name: 'assistant', model: 'fixture', systemPrompt: 'test' }],
      modules: [new CaptureContext('capture', resources), new StartupResource('failure', resources, failure)],
      syncIntervalMs: 0,
    }), error => error === failure);
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

for (const callerOwned of [false, true]) {
  for (const cleanupFails of [false, true]) {
    test(`failed create cleans partial modules before store reopen (caller owned=${callerOwned}, cleanup fails=${cleanupFails})`, async () => {
      const dir = mkdtempSync(join(tmpdir(), 'af-create-cleanup-'));
      const storePath = join(dir, 'store');
      const store = callerOwned ? JsStore.openOrCreate({ path: storePath }) : undefined;
      const resources = new Set<MessagePort>();
      const startError = new Error('second module initialization failed');
      const stopError = cleanupFails ? new Error('first module cleanup failed') : undefined;
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
