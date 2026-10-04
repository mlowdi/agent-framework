import { normalizeImageContent, projectNativeImageContent, isVisualImageContent, type ContentBlock } from '@animalabs/membrane';
import { INLINE_WITHHELD_TEXT, isInlineContradiction, referenceStubOrNull } from './mcpl/references.js';

/** Project structured tool/recall data in traversal order without ever placing
 * image payloads in JSON text. Unknown metadata remains readable, while raw
 * native carriers stay in the audit rather than being duplicated into text.
 * null preserves the existing JSON/text-only behavior for non-media results. */
export function projectToolResultMedia(data: unknown): ContentBlock[] | null {
  let hasMedia = false;
  const blocks: ContentBlock[] = [];
  const pending: string[] = [];
  const flush = (): void => {
    if (pending.length) { blocks.push({ type: 'text', text: pending.join('') }); pending.length = 0; }
  };
  const walk = (value: unknown): void => {
    if (Array.isArray(value)) {
      // Content arrays may contain legacy text-only rawItem projections.
      const isContent = value.every(b => b && typeof b === 'object' && typeof b.type === 'string');
      const parts = isContent ? projectNativeImageContent({ content: value as ContentBlock[] }) : value;
      if (!isContent) pending.push('[');
      for (let i = 0; i < parts.length; i++) {
        if (i > 0 && !isContent) pending.push(',');
        walk(parts[i]);
      }
      if (!isContent) pending.push(']');
      return;
    }
    if (!value || typeof value !== 'object') { pending.push(JSON.stringify(value) ?? 'null'); return; }
    const block = value as Record<string, unknown>;
    if (isInlineContradiction(block)) { hasMedia = true; pending.push(INLINE_WITHHELD_TEXT); return; }
    if (block.type === 'resource' || ((block.type === 'image' || block.type === 'audio') && typeof block.uri === 'string')) {
      hasMedia = true;
      pending.push(referenceStubOrNull(block, 'from tool result') ?? '[reference unavailable: invalid source]');
      return;
    }
    if (isVisualImageContent(block) || block.type === 'input_image') {
      hasMedia = true;
      flush();
      const image = normalizeImageContent(block);
      // Generated producer metadata and native originals remain archival
      // testimony. Formatters project the visual at their own wire boundary.
      if (image.type === 'generated_image') blocks.push(image);
      else if (image.type === 'image') {
        const { rawItem: _rawItem, ...visual } = image;
        blocks.push(visual);
      } else blocks.push(image);
      return;
    }
    if (block.type === 'text' && typeof block.text === 'string') {
      if (block.rawItem) hasMedia = true;
      flush();
      blocks.push({ type: 'text', text: block.text });
      return;
    }
    pending.push('{');
    let fields = 0;
    for (const [key, child] of Object.entries(block)) {
      if (fields++ > 0) pending.push(',');
      pending.push(JSON.stringify(key), ':');
      if (key === 'rawItem') {
        hasMedia = true;
        pending.push('"[provider-native original retained in archive]"');
      } else walk(child);
    }
    pending.push('}');
  };
  walk(data);
  flush();
  return hasMedia ? blocks : null;
}
