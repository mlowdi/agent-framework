import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { Membrane, OpenAIResponsesFormatter, type ProviderAdapter, type ProviderRequest,
  type ProviderResponse, type StreamCallbacks, type ContentBlock } from '@animalabs/membrane';
import { ContextManager, AutobiographicalStrategy } from '@animalabs/context-manager';
import { AgentFramework } from '../src/index.js';
import type { Module, ModuleContext, ProcessEvent, ProcessState, ToolCall, ToolResult } from '../src/types/index.js';
import { TOOL_RESULT_GUARD_AUDIT_STATE } from '../src/tool-result-guard.js';
import { ToolImageLedger, parseImagePlaceholders } from '../src/tool-image-ledger.js';

const PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==';
class WireProvider implements ProviderAdapter {
  readonly name = 'framework-media-fixture';
  readonly frames: ProviderRequest[] = [];
  constructor(private tools: string[]) {}
  supportsModel(): boolean { return true; }
  async complete(request: ProviderRequest): Promise<ProviderResponse> {
    this.frames.push(request);
    return this.reply();
  }
  async stream(request: ProviderRequest, callbacks: StreamCallbacks): Promise<ProviderResponse> {
    this.frames.push(request);
    const response = this.reply();
    response.content.forEach((block, index) => {
      callbacks.onContentBlock?.(index, block);
      if (block.type === 'text' && typeof block.text === 'string') callbacks.onChunk(block.text);
    });
    return response;
  }
  private reply(): ProviderResponse & { content: ContentBlock[] } {
    const tool = this.tools.shift();
    const request = this.frames[this.frames.length - 1]!;
    const content: ContentBlock[] = tool ? [{ type: 'tool_use', id: tool, name: `media--${tool}`, input: {} }]
      : [{ type: 'text', text: 'archive-visible answer' }];
    return { content, model: request.model, rawRequest: request, stopReason: tool ? 'tool_use' : 'end_turn',
      usage: { inputTokens: 40, outputTokens: 10 }, raw: {} };
  }
}

class MediaModule implements Module {
  readonly name = 'media';
  private context!: ModuleContext;
  async start(context: ModuleContext) { this.context = context; }
  async stop() {}
  getTools() { return ['inspect', 'recall', 'invalid', 'error'].map(name => ({
    name, description: name, inputSchema: { type: 'object' as const, properties: {} },
  })); }
  async handleToolCall(call: ToolCall): Promise<ToolResult> {
    if (call.name === 'recall') return { success: true, data: {
      messages: this.context.queryMessages({}).messages.filter(message => message.content.some(block =>
        block.type === 'tool_result' && block.toolUseId === 'inspect')),
    } };
    if (call.name === 'invalid') return { success: true, data: { messages: [{ content: [
      { type: 'text', text: 'before invalid' }, { type: 'image', data: PNG },
      { type: 'image', source: { type: 'base64', data: PNG, mediaType: 'image/svg+xml' } },
      { type: 'text', text: 'after invalid' },
    ] }] } };
    return { success: call.name !== 'error', ...(call.name === 'error' ? { isError: true, error: 'inspection failed' } : {}),
      data: [{ type: 'text', text: 'before' }, { type: 'image', data: PNG, mimeType: 'image/png' },
        { type: 'text', text: 'between' }, { type: 'image', source: { type: 'url', url: 'https://example.test/current.png' } },
        { type: 'text', text: 'after' }] };
  }
  async onProcess(event: ProcessEvent, _state: ProcessState) {
    return event.type === 'external-message' ? { addMessages: [{ participant: 'user',
      content: [{ type: 'text' as const, text: 'inspect it' }] }], requestInference: true } : {};
  }
}
function wireImages(request: ProviderRequest): string[] {
  const images: string[] = [];
  const walk = (value: unknown): void => {
    if (Array.isArray(value)) { value.forEach(walk); return; }
    if (!value || typeof value !== 'object') return;
    if ('type' in value && value.type === 'input_image' && 'image_url' in value) images.push(String(value.image_url));
    Object.values(value).forEach(walk);
  };
  walk(request.messages);
  return images;
}
function wireText(request: ProviderRequest): string {
  const text: string[] = [];
  const walk = (value: unknown): void => {
    if (Array.isArray(value)) { value.forEach(walk); return; }
    if (!value || typeof value !== 'object') return;
    if ('type' in value && value.type === 'input_text' && 'text' in value) text.push(String(value.text));
    Object.values(value).forEach(walk);
  };
  walk(request.messages);
  return text.join('\n');
}
const policyStrategy = () => new AutobiographicalStrategy({ headWindowTokens: 0, recentWindowTokens: 100000,
  targetChunkTokens: 100000, hierarchical: true, autoTickOnNewMessage: false,
  maxLiveImages: 2, maxLiveImageBytes: PNG.length, imageStripDepthTokens: 0 });

test('framework media, raw recall, recompile and restart keep images typed and inventory bytes retrievable', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'af-media-continuity-'));
  const storePath = join(dir, 'store');
  const provider = new WireProvider(['inspect', 'recall']);
  const membrane = new Membrane(provider, { formatter: new OpenAIResponsesFormatter() });
  const framework = await AgentFramework.create({ storePath, membrane,
    agents: [{ name: 'assistant', model: 'fixture', systemPrompt: 'inspect',
      strategy: policyStrategy(), toolResultGuard: false }], modules: [new MediaModule()], syncIntervalMs: 0 });
  let stopped = false;
  let reopened: ContextManager | undefined;
  try {
    framework.pushEvent({ type: 'external-message', source: 'headless', content: 'inspect', metadata: {} });
    await framework.runUntilIdle();
    assert.equal(provider.frames.length, 3, 'two real tool rounds reached the native request boundary');
    assert.deepEqual(wireImages(provider.frames[1]), [`data:image/png;base64,${PNG}`, 'https://example.test/current.png']);
    assert.deepEqual(wireImages(provider.frames[2]), [`data:image/png;base64,${PNG}`, 'https://example.test/current.png'],
      'raw recall projected nested media and accounted for already-retained images');
    for (const frame of provider.frames) assert.ok(!wireText(frame).includes(PNG));
    const cm = framework.getAgent('assistant')!.getContextManager();
    const results = cm.getAllMessages().flatMap(message => message.content).filter(block => block.type === 'tool_result');
    const inspected = results.find(block => block.type === 'tool_result' && block.toolUseId === 'inspect');
    assert.ok(inspected?.type === 'tool_result' && Array.isArray(inspected.content));
    assert.deepEqual(inspected.content.map(block => block.type), ['text', 'image', 'text', 'image', 'text']);
    const refs = parseImagePlaceholders(inspected.imageHistory!);
    assert.equal(refs.length, 1, 'the base64 inventory is independent of the visual archive');
    const first = refs[0];
    assert.ok(first && 'ref' in first && first.ref);
    const ref = first.ref;
    const compiled = await cm.compile({ maxTokens: 100000, reserveForResponse: 0 });
    await membrane.complete({ messages: compiled.messages, config: { model: 'fixture', maxTokens: 100 }, toolMode: 'native',
      liveImagePolicy: cm.getLiveImagePolicy() });
    assert.equal(wireImages(provider.frames.at(-1)!).length, 2);
    assert.ok(!wireText(provider.frames.at(-1)!).includes(PNG));
    await framework.stop(); stopped = true;
    reopened = await ContextManager.open({ path: storePath, namespace: 'agents/assistant', strategy: policyStrategy() });
    const restored = reopened.getAllMessages().flatMap(message => message.content).find(block =>
      block.type === 'tool_result' && block.toolUseId === 'inspect');
    assert.ok(restored?.type === 'tool_result' && Array.isArray(restored.content));
    const retained = restored.content.find(block => block.type === 'image' && block.source.type === 'base64');
    assert.ok(retained?.type === 'image' && retained.source.type === 'base64');
    assert.equal(retained.source.data, PNG);
    const ledger = new ToolImageLedger({ store: reopened.getStore(), stateId: 'framework/tool-images/assistant', maxEntries: 1 });
    const recovered = ledger.lookup(ref);
    assert.equal(recovered.status, 'retained', 'original bytes survive process-cache eviction and restart');
    if (recovered.status === 'retained') assert.equal(recovered.image.data, PNG);
  } finally { reopened?.close(); if (!stopped) await framework.stop(); rmSync(dir, { recursive: true, force: true }); }
});

for (const tool of ['invalid', 'error']) {
  test(`${tool} image results stay bounded/visual as appropriate and their original shapes remain archived`, async () => {
    const dir = mkdtempSync(join(tmpdir(), 'af-media-invalid-'));
    const provider = new WireProvider([tool]);
    const framework = await AgentFramework.create({ storePath: join(dir, 'store'),
      membrane: new Membrane(provider, { formatter: new OpenAIResponsesFormatter() }),
      agents: [{ name: 'assistant', model: 'fixture', systemPrompt: 'inspect', strategy: policyStrategy() }],
      modules: [new MediaModule()], syncIntervalMs: 0 });
    try {
      framework.pushEvent({ type: 'external-message', source: 'headless', content: 'inspect', metadata: {} });
      await framework.runUntilIdle();
      assert.equal(provider.frames.length, 2);
      assert.equal(wireImages(provider.frames[1]).length, tool === 'invalid' ? 0 : 2);
      assert.ok(!wireText(provider.frames[1]).includes(PNG));
      const cm = framework.getAgent('assistant')!.getContextManager();
      const audit = cm.getStore().getStateJson(TOOL_RESULT_GUARD_AUDIT_STATE);
      assert.ok(JSON.stringify(audit).includes(PNG), 'even unguarded/invalid/error originals remain independently retrievable');
      const result = cm.getAllMessages().flatMap(message => message.content).find(block =>
        block.type === 'tool_result' && block.toolUseId === tool);
      assert.equal(result?.type === 'tool_result' && result.isError === true, tool === 'error');
    } finally { await framework.stop(); rmSync(dir, { recursive: true, force: true }); }
  });
}
