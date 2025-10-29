import runpod
import time  
import os
from llama_cpp import Llama
import re


volume_path = '/runpod-volume'
found_nas = os.path.isdir(volume_path)
found_gguf = False
gguf_path = None
if found_nas:
    print(f"Found NAS at {volume_path}, looking for ggufs...")
    try:
        # Loop through all items in the confirmed directory
        for item in os.listdir(volume_path):
            if item.lower().endswith('.gguf'):
                full_path = os.path.join(volume_path, item)
                if os.path.isfile(full_path):
                    gguf_path = full_path
                    found_gguf = True
    except Exception as e:
        # Catch other potential errors, like read permissions
        print(f"An error occurred accessing the volume: {str(e)}")
else:
    print("No NAS found")


if found_gguf:
    print(f"Found gguf model at {gguf_path}, loading...")
    llm = Llama(
        model_path=gguf_path,
        n_ctx=16384,      # The context size for the model.
        n_gpu_layers=-1, # Offload all layers to the GPU. Set to 0 for CPU-only.
        verbose=False,    # Enable verbose logging from llama.cpp.
    )




# small helper function for removing special token stuff
def clean_stream(stream):
    buffer = ""
    current_channel = None
    analysis_open = False

    for chunk in stream:
        content = chunk["choices"][0].get("delta", {}).get("content")
        if content is None:
            continue
        buffer += content

        while True:
            if current_channel is None:
                # Look for a full opener: <|channel|>NAME<|message|>
                m = re.search(r"<\|channel\|>([^<]*)<\|message\|>", buffer)
                if not m:
                    # No complete opener yet — keep buffer (it may contain partial opener).
                    # Don't emit anything before we know the channel.
                    # If buffer grows huge, trim keeping a little tail to preserve split markers.
                    if len(buffer) > 4096:
                        buffer = buffer[-4096:]
                    break

                # Discard any garbage before the opener
                if m.start() > 0:
                    buffer = buffer[m.start():]
                    m = re.search(r"<\|channel\|>([^<]*)<\|message\|>", buffer)
                    if not m:
                        break

                current_channel = m.group(1).strip()
                buffer = buffer[m.end():]  # remove the opener from buffer

                if current_channel == "analysis":
                    # Immediately emit opening tag for analysis
                    yield "<t>"
                    analysis_open = True

                # Continue to process body (don't break)
            else:
                # We have an active channel; stream content up to next marker if any.
                next_marker = buffer.find("<|")
                if next_marker == -1:
                    # No marker seen in buffer. Emit everything except a single trailing '<' (to handle split marker).
                    if buffer:
                        tail = ""
                        if buffer.endswith("<"):
                            tail = "<"
                            to_emit = buffer[:-1]
                        else:
                            to_emit = buffer
                        if to_emit:
                            yield to_emit
                        buffer = tail
                    break

                # Found a marker start: emit content up to marker, then prepare to parse the marker
                message = buffer[:next_marker]
                if message:
                    yield message
                buffer = buffer[next_marker:]  # keep the marker start for next loop

                # Close analysis wrapper if needed before parsing next opener
                if analysis_open:
                    yield "</t>"
                    analysis_open = False

                current_channel = None
                # loop continues to parse the marker

    # Stream ended: flush remaining content
    if current_channel is not None and buffer:
        # If there's a remaining body for the active channel, emit it
        yield buffer
        buffer = ""

    # If analysis tag is still open, close it
    if analysis_open:
        yield "</t>"

async def handler(event):
    global found_nas, found_gguf, llm
    print(f"Worker Start")
    if not found_nas:
        return "No NAS found on startup"
    if not found_gguf:
        return "No gguf model found on startup"
    input = event['input']

    prompt = input.get('prompt')  

    print(f"Received prompt: {prompt}")
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    stream = llm.create_chat_completion(
        messages=messages,
        stream=True
    )
    for token in clean_stream(stream):
        yield token

    return

if __name__ == '__main__':
    runpod.serverless.start({
        'handler': handler,
        "return_aggregate_stream": True
    })
