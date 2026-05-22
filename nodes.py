# --- START OF FILE nodes.py ---

import logging
import json
import time
import os
import folder_paths

from comfy_api.latest import io

from .prompt_relay import (
    get_raw_tokenizer,
    map_token_indices,
    build_segments,
    create_mask_fn,
    distribute_segment_lengths,
)
from .patches import detect_model_type, apply_patches
from .advanced_options import PromptRelayAdvancedOptions, RelayOptions

log = logging.getLogger(__name__)

def _format_prompts_to_text(global_prompt, local_prompts, segment_lengths, epsilon, max_frames=None, fps=24.0, timeline_data="{}"):
    lines = []
    lines.append("Prompt Relay Export")
    lines.append("===================")
    lines.append("")
    lines.append("=== Global Parameters ===")
    lines.append(f"Global Prompt:\n{global_prompt}\n")
    
    fr = float(fps) if fps else 24.0
    if fr <= 0: fr = 24.0
    
    if max_frames is not None:
        lines.append(f"Duration: {max_frames} frames ({(max_frames / fr):.2f}s @ {fr} FPS)")
    
    lines.append(f"Epsilon (Penalty Decay): {epsilon}")
    lines.append("")
    lines.append("=== Timeline Segments ===")
    
    locals_list = [p.strip() for p in local_prompts.split("|")] if local_prompts else []
    lengths_list = [l.strip() for l in segment_lengths.split(",")] if segment_lengths else []
    
    if locals_list and any(locals_list):
        lines.append("\n--- Text Prompts ---")
        current_frame = 0.0
        for i, prompt in enumerate(locals_list):
            try:
                len_f = float(lengths_list[i]) if i < len(lengths_list) and lengths_list[i] else 0.0
            except ValueError:
                len_f = 0.0
            
            start_f = current_frame
            end_f = start_f + len_f
            
            start_s = start_f / fr
            end_s = end_f / fr
            len_s = len_f / fr
            
            lines.append(f"\n[Prompt {i+1}]")
            lines.append(f"Time: {start_s:.2f}s - {end_s:.2f}s (Duration: {len_s:.2f}s)")
            lines.append(f"Frames: {start_f:.1f} - {end_f:.1f} (Length: {len_f:.1f})")
            lines.append(f"Prompt:\n{prompt}")
            lines.append("-" * 40)
            
            current_frame += len_f
            
    return "\n".join(lines)


# Register API endpoint for instant export from the JS UI if needed
try:
    import server
    from aiohttp import web

    @server.PromptServer.instance.routes.post("/prompt_relay/export_timeline")
    async def export_prompt_relay_endpoint(request):
        try:
            data = await request.json()
            global_prompt = data.get("global_prompt", "")
            local_prompts = data.get("local_prompts", "")
            segment_lengths = data.get("segment_lengths", "")
            epsilon = float(data.get("epsilon", 0.001))
            
            max_frames = data.get("max_frames", None)
            if max_frames is not None:
                max_frames = int(max_frames)
                
            fps = float(data.get("fps", 24.0))
            timeline_data = data.get("timeline_data", "{}")
            
            formatted_text = _format_prompts_to_text(
                global_prompt, local_prompts, segment_lengths, epsilon, max_frames, fps, timeline_data
            )
            
            out_dir = folder_paths.get_output_directory()
            filename = f"prompt_relay_prompts_{int(time.time())}.txt"
            filepath = os.path.join(out_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(formatted_text)
                
            return web.json_response({
                "status": "success", 
                "filepath": filepath, 
                "filename": filename, 
                "content": formatted_text
            })
        except Exception as e:
            log.error(f"[PromptRelay] Export endpoint error: {e}")
            return web.json_response({"status": "error", "message": str(e)}, status=500)
except Exception as e:
    log.warning(f"[PromptRelay] Could not register /prompt_relay/export_timeline endpoint: {e}")


def _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames):
    if not pixel_lengths:
        return[]
    total_pixel = sum(pixel_lengths)
    if total_pixel <= 0:
        return [1] * len(pixel_lengths)

    naive_total = max(1, round(total_pixel / temporal_stride))
    target_total = min(latent_frames, naive_total)
    if target_total >= latent_frames - 1:
        target_total = latent_frames

    exact = [p * target_total / total_pixel for p in pixel_lengths]
    result = [int(e) for e in exact]
    diff = target_total - sum(result)
    if diff > 0:
        order = sorted(range(len(exact)), key=lambda i: -(exact[i] - int(exact[i])))
        for k in range(diff):
            result[order[k % len(order)]] += 1

    for i in range(len(result)):
        if result[i] < 1:
            max_idx = max(range(len(result)), key=lambda j: result[j])
            if result[max_idx] > 1:
                result[max_idx] -= 1
                result[i] = 1

    return result

def _encode_relay(model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon, relay_options=None):
    for name, val in (("global_prompt", global_prompt),
                      ("local_prompts", local_prompts),
                      ("segment_lengths", segment_lengths)):
        if val is None:
            raise ValueError(f"PromptRelay: '{name}' arrived as None.")

    locals_list = [p.strip() for p in local_prompts.split("|") if p.strip()]
    if not locals_list:
        raise ValueError("At least one local prompt is required (separate with |)")

    arch, patch_size, temporal_stride = detect_model_type(model)
    samples = latent["samples"]
    latent_frames = samples.shape[2]
    tokens_per_frame = (samples.shape[3] // patch_size[1]) * (samples.shape[4] // patch_size[2])

    parsed_lengths = None
    if segment_lengths.strip():
        pixel_lengths = [int(x.strip()) for x in segment_lengths.split(",") if x.strip()]
        parsed_lengths = _convert_to_latent_lengths(pixel_lengths, temporal_stride, latent_frames)

    raw_tokenizer = get_raw_tokenizer(clip)
    full_prompt, token_ranges = map_token_indices(raw_tokenizer, global_prompt, locals_list)
    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(full_prompt))

    effective_lengths = distribute_segment_lengths(len(locals_list), latent_frames, parsed_lengths)
    q_token_idx = build_segments(token_ranges, effective_lengths, epsilon, relay_options)
    mask_fn = create_mask_fn(q_token_idx, tokens_per_frame, latent_frames)

    patched = model.clone()
    apply_patches(patched, arch, mask_fn)

    return patched, conditioning

class PromptRelayEncode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PromptRelayEncode",
            display_name="Prompt Relay Encode",
            category="conditioning/prompt_relay",
            description="Encodes a global prompt combined with temporal local prompts.",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Latent.Input("latent"),
                io.String.Input("global_prompt", multiline=True, default=""),
                io.String.Input("local_prompts", multiline=True, default=""),
                io.String.Input("segment_lengths", default=""),
                io.Float.Input("epsilon", default=1e-3, min=1e-6, max=0.99, step=1e-4),
                io.Boolean.Input("save_prompts_to_file", default=False, optional=True, tooltip="Save the timeline prompts and parameters to a text file in your ComfyUI output directory during execution."),
                RelayOptions.Input("relay_options", optional=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon, save_prompts_to_file=False, relay_options=None) -> io.NodeOutput:
        patched, conditioning = _encode_relay(
            model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon, relay_options,
        )

        if save_prompts_to_file:
            try:
                formatted_text = _format_prompts_to_text(
                    global_prompt, local_prompts, segment_lengths, epsilon, max_frames=None, fps=24.0
                )
                out_dir = folder_paths.get_output_directory()
                filename = f"prompt_relay_{int(time.time())}.txt"
                filepath = os.path.join(out_dir, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(formatted_text)
                log.info(f"[PromptRelay] Saved prompts to {filepath}")
            except Exception as e:
                log.warning(f"[PromptRelay] Failed to save prompts to txt: {e}")

        return io.NodeOutput(patched, conditioning)

class PromptRelayEncodeTimeline(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="PromptRelayEncodeTimeline",
            display_name="Prompt Relay Encode (Timeline)",
            category="conditioning/prompt_relay",
            description="Features an ON/OFF switch to safely sync the LTX Sequencer without messy wires.",
            inputs=[
                io.Model.Input("model"),
                io.Clip.Input("clip"),
                io.Latent.Input("latent", tooltip="Empty latent video — dimensions are read from its shape."),
                io.String.Input("global_prompt", multiline=True, default=""),
                io.Int.Input("max_frames", default=129, min=1, max=10000, step=1),
                io.String.Input("timeline_data", default=""),
                io.String.Input("local_prompts", multiline=True, default=""),
                io.String.Input("segment_lengths", default=""),
                io.Float.Input("epsilon", default=1e-3, min=1e-6, max=0.99, step=1e-4),
                io.Float.Input("fps", default=24.0, min=0.1, max=240.0, step=0.1, optional=True),
                io.Combo.Input("time_units", options=["frames", "seconds"], default="frames", optional=True),
                # ---> NEW TOGGLE SWITCH <---
                io.Combo.Input("sequencer_sync", options=["ON", "OFF"], default="ON", tooltip="Turn ON to auto-sync the LTX Sequencer. Turn OFF to use manual sliders."),
                io.Boolean.Input("save_prompts_to_file", default=False, optional=True, tooltip="Save the timeline prompts and parameters to a text file in your ComfyUI output directory during execution."),
                RelayOptions.Input("relay_options", optional=True),
            ],
            outputs=[
                io.Model.Output(display_name="model"),
                io.Conditioning.Output(display_name="positive"),
            ],
        )

    @classmethod
    def execute(cls, model, clip, latent, global_prompt, max_frames, timeline_data, local_prompts, segment_lengths, epsilon, fps=24.0, time_units="frames", sequencer_sync="ON", save_prompts_to_file=False, relay_options=None) -> io.NodeOutput:
        patched, conditioning = _encode_relay(
            model, clip, latent, global_prompt, local_prompts, segment_lengths, epsilon, relay_options,
        )

        if save_prompts_to_file:
            try:
                formatted_text = _format_prompts_to_text(
                    global_prompt, local_prompts, segment_lengths, epsilon, max_frames, fps, timeline_data
                )
                out_dir = folder_paths.get_output_directory()
                filename = f"prompt_relay_timeline_{int(time.time())}.txt"
                filepath = os.path.join(out_dir, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(formatted_text)
                log.info(f"[PromptRelay] Saved timeline prompts to {filepath}")
            except Exception as e:
                log.warning(f"[PromptRelay] Failed to save prompts to txt: {e}")

        if sequencer_sync == "ON":
            parsed_lengths = []
            if segment_lengths and segment_lengths.strip():
                parsed_lengths =[int(x.strip()) for x in segment_lengths.split(",") if x.strip()]
            
            starts_frames =[]
            current = 0
            for length in parsed_lengths:
                starts_frames.append(current)
                current += length
                
            timeline_info = {
                "starts_frames": starts_frames,
                "starts_seconds":[f / fps for f in starts_frames] if fps else [0]*len(starts_frames),
                "fps": fps
            }
            timeline_json = json.dumps(timeline_info)

            # Safely inject invisible sync data into the yellow positive wire!
            for c in conditioning:
                c[1]["prompt_relay_timeline"] = timeline_json

        return io.NodeOutput(patched, conditioning)

NODE_CLASS_MAPPINGS = {
    "PromptRelayEncode": PromptRelayEncode,
    "PromptRelayEncodeTimeline": PromptRelayEncodeTimeline,
    "PromptRelayAdvancedOptions": PromptRelayAdvancedOptions,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptRelayEncode": "Prompt Relay Encode",
    "PromptRelayEncodeTimeline": "Prompt Relay Encode (Timeline)",
    "PromptRelayAdvancedOptions": "Prompt Relay Advanced Options",
}
