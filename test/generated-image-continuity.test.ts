import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Membrane, OpenAIResponsesFormatter } from '@animalabs/membrane';
import type { ContentBlock, GeneratedImageContent, ProviderAdapter, ProviderRequest, ProviderResponse, StreamCallbacks } from '@animalabs/membrane';
import { ContextManager, AutobiographicalStrategy } from '@animalabs/context-manager';
import { AgentFramework } from '../src/index.js';
import type { Module, ModuleContext, ProcessEvent, ToolCall, ToolResult, TraceEvent } from '../src/types/index.js';
import { ToolImageLedger, parseImagePlaceholders } from '../src/tool-image-ledger.js';

const PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';
const native = { type: 'image_generation_call', id: 'generated-final', status: 'completed', result: PNG, output_format: 'png' };
const output: ContentBlock[] = [{ type: 'text', text: 'before output' },
  { type: 'generated_image', data: PNG, mimeType: 'image/png', tokenEstimate: 0, isPreview: true,
    rawItem: { type: 'foreign-native', producer: 'unchanged' } }, { type: 'text', text: 'between output' },
  { type: 'generated_image', data: PNG, mimeType: 'image/png', tokenEstimate: 731, isPreview: false, rawItem: native },
  { type: 'text', text: 'after output' }];
const strategy = () => new AutobiographicalStrategy({ headWindowTokens: 0, recentWindowTokens: 100000,
  targetChunkTokens: 100000, adaptiveResolution: false, autoTickOnNewMessage: false,
  maxLiveImages: 0, maxLiveImageBytes: 0, imageStripDepthTokens: 0 });
class GeneratedProvider implements ProviderAdapter {
  readonly name = 'framework-generated-fixture';
  readonly frames: ProviderRequest[] = [];
  supportsModel() { return true; }
  async complete(request: ProviderRequest): Promise<ProviderResponse> {
    this.frames.push(request);
    const first = this.frames.length === 1;
    return { content: first ? [{ type: 'tool_use', id: 'generate-fixture', name: 'fixture--inspect', input: {} }] : output,
      model: request.model, rawRequest: request, stopReason: first ? 'tool_use' : 'end_turn', raw: {}, usage: { inputTokens: 20, outputTokens: 5 } };
  }
  async stream(request: ProviderRequest, callbacks: StreamCallbacks): Promise<ProviderResponse> {
    const response = await this.complete(request);
    (response.content as ContentBlock[]).forEach((block, index) => {
      callbacks.onContentBlock?.(index, block);
      if (block.type === 'text') callbacks.onChunk(block.text);
    });
    return response;
  }
}
class GeneratedToolModule implements Module {
  readonly name = 'fixture';
  async start(_context: ModuleContext) {}
  async stop() {}
  getTools() { return [{ name: 'inspect', description: 'synthetic generated media, no generation API', inputSchema: { type: 'object' as const, properties: {} } }]; }
  async handleToolCall(_call: ToolCall): Promise<ToolResult> {
    return { success: false, isError: true, error: 'fixture error with retained media', data: {
      caption: 'before tool', nested: [{ type: 'generated_image', data: PNG, mimeType: 'image/png', tokenEstimate: 0, isPreview: true },
        { type: 'tool_result', toolUseId: 'nested', content: [{ type: 'text', text: 'between tool' },
          { type: 'generated_image', data: PNG, mimeType: 'image/png', tokenEstimate: 907, isPreview: false },
          { type: 'text', text: 'after tool' }] }], trace: 'unrelated field survives',
    } };
  }
  async onProcess(event: ProcessEvent) {
    return event.type === 'external-message' ? { addMessages: [{ participant: 'user', content: [{ type: 'text' as const, text: 'inspect fixture' }] }], requestInference: true } : {};
  }
}
function wireParts(value: unknown): Record<string, unknown>[] {
  if (Array.isArray(value)) return value.flatMap(wireParts);
  if (!value || typeof value !== 'object') return [];
  return [value as Record<string, unknown>, ...Object.values(value).flatMap(wireParts)];
}

test('actual framework output/non-MCP/error ingestion archives generated media, replays after reopen and preserves ledger originals', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'af-generated-continuity-'));
  const storePath = join(dir, 'store');
  const provider = new GeneratedProvider();
  const membrane = new Membrane(provider, { formatter: new OpenAIResponsesFormatter() });
  const framework = await AgentFramework.create({ storePath, membrane, agents: [{ name: 'assistant', model: 'fixture',
    systemPrompt: 'inspect', strategy: strategy(), toolResultGuard: false }], modules: [new GeneratedToolModule()], syncIntervalMs: 0 });
  let stopped = false;
  let reopened: ContextManager | undefined;
  try {
    framework.pushEvent({ type: 'external-message', source: 'fixture', content: 'inspect', metadata: {} });
    await framework.runUntilIdle();
    assert.equal(provider.frames.length, 2);
    const toolWire = wireParts(provider.frames[1].messages);
    assert.equal(toolWire.filter(part => part.type === 'input_image').length, 2);
    assert.ok(!toolWire.some(part => typeof part.text === 'string' && part.text.includes(PNG)));
    const cm = framework.getAgent('assistant')!.getContextManager();
    const all = cm.getAllMessages();
    const result = all.flatMap(message => message.content).find(block => block.type === 'tool_result' && block.toolUseId === 'generate-fixture');
    assert.ok(result?.type === 'tool_result' && Array.isArray(result.content));
    assert.equal(result.isError, true);
    assert.deepEqual(result.content.filter(block => block.type === 'generated_image').map(block => block.tokenEstimate), [0, 907]);
    assert.ok(result.content.some(block => block.type === 'text' && block.text.includes('unrelated field survives')));
    const assistant = all.find(message => message.content.some(block => block.type === 'generated_image' && (block.rawItem as { type?: string } | undefined)?.type === 'image_generation_call'))!;
    assert.deepEqual(assistant.content, output);
    const inventory = parseImagePlaceholders(result.imageHistory!);
    assert.equal(inventory.length, 2);
    const refs = inventory.map(slot => 'ref' in slot ? slot.ref : null);
    assert.ok(refs.every(Boolean));
    const compiled = await cm.compile({ maxTokens: 100000, reserveForResponse: 0 });
    await membrane.complete({ messages: compiled.messages, config: { model: 'fixture', maxTokens: 100 }, toolMode: 'native',
      assistantParticipant: 'assistant', liveImagePolicy: cm.getLiveImagePolicy() });
    const replay = wireParts(provider.frames.at(-1)!.messages);
    assert.equal(replay.filter(part => part.type === 'input_image').length, 3, 'two tool visuals plus one explicit-origin preview companion');
    assert.equal(replay.filter(part => part.type === 'image_generation_call').length, 1, 'native final is authoritative, not a second companion');
    assert.ok(replay.some(part => typeof part.text === 'string' && part.text.includes('not user authorship or a new user request')));
    await framework.stop(); stopped = true;
    reopened = await ContextManager.open({ path: storePath, namespace: 'agents/assistant', strategy: strategy() });
    const restored = reopened.getAllMessages();
    assert.deepEqual(restored.find(message => message.id === assistant.id)!.content, output);
    const restoredResult = restored.flatMap(message => message.content).find(block => block.type === 'tool_result' && block.toolUseId === 'generate-fixture');
    assert.ok(restoredResult?.type === 'tool_result' && Array.isArray(restoredResult.content));
    assert.deepEqual(restoredResult.content.filter(block => block.type === 'generated_image').map(block => (block as GeneratedImageContent).data), [PNG, PNG]);
    const ledger = new ToolImageLedger({ store: reopened.getStore(), stateId: 'framework/tool-images/assistant', maxEntries: 1 });
    for (const ref of refs) {
      const recovered = ledger.lookup(ref!);
      assert.equal(recovered.status, 'retained');
      if (recovered.status === 'retained') assert.equal(recovered.image.data, PNG);
    }
  } finally { reopened?.close(); if (!stopped) await framework.stop(); rmSync(dir, { recursive: true, force: true }); }
});

test('falsifier: process diagnostics redact generated and canonical payloads without changing authoritative addMessages', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'af-generated-process-diagnostics-'));
  const content: ContentBlock[] = [{ type: 'text', text: 'before' },
    { type: 'generated_image', data: PNG, mimeType: 'image/png', tokenEstimate: 0, isPreview: true, rawItem: native },
    { type: 'image', source: { type: 'base64', data: PNG, mediaType: 'image/png' }, tokenEstimate: 731, rawItem: native },
    { type: 'text', text: 'after' }];
  const module: Module = { name: 'diagnostic-fixture', async start() {}, async stop() {}, getTools() { return []; },
    async handleToolCall() { return { success: false, error: 'no fixture tools' }; },
    async onProcess(event) { return event.type === 'external-message'
      ? { addMessages: [{ participant: 'user', content }], requestInference: false } : {}; },
  };
  const provider = new GeneratedProvider();
  const framework = await AgentFramework.create({ storePath: join(dir, 'store'), membrane: new Membrane(provider),
    agents: [{ name: 'assistant', model: 'unused-no-model', systemPrompt: '', strategy: strategy() }], modules: [module],
    processLogging: { persist: true, broadcast: true }, syncIntervalMs: 0 });
  const traces: TraceEvent[] = [];
  const detach = framework.onTrace(event => traces.push(event));
  try {
    framework.pushEvent({ type: 'external-message', source: 'diagnostic-fixture', content: 'synthetic fixture', metadata: {} });
    await framework.runUntilIdle();
    const trace = traces.find(event => event.type === 'process:completed');
    assert.ok(trace, 'actual process-completed trace emitted');
    assert.ok(!JSON.stringify(trace).includes(PNG), 'trace sent to host must be a payload-free diagnostic view');
    const log = framework.getProcessLog(0);
    assert.ok(log, 'actual persisted process diagnostic exists');
    assert.ok(!JSON.stringify(log).includes(PNG), 'persisted diagnostic must not inline binary/native carriers');
    const cm = framework.getAgent('assistant')!.getContextManager();
    const archived = cm.getAllMessages().find(message => message.content.some(block => block.type === 'generated_image'))!;
    assert.deepEqual(archived.content, content, 'authoritative message and native carriers remain untouched');
    assert.equal(provider.frames.length, 0, 'no provider/model call');
    assert.equal(content[1].rawItem, native);
  } finally { detach(); await framework.stop(); rmSync(dir, { recursive: true, force: true }); }
});
