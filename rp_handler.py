import runpod
import time  
import os
from llama_cpp import Llama


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
        verbose=True,    # Enable verbose logging from llama.cpp.
    )

def handler(event):
    global found_nas, found_gguf, llm
    print(f"Worker Start")
    if not found_nas:
        yield "No NAS found on startup"
        return
    if not found_gguf:
        yield "No gguf model found on startup"
        return # I think both yields could be just returns but lets be safe
    input = event['input']

    prompt = input.get('prompt')  
    seconds = input.get('seconds', 0)  

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
    for chunk in stream:
        content = chunk["choices"][0].get("delta", {}).get("content")
        if content is not None:
            yield content

    print(f"Sleeping for {seconds} seconds...")
    
    # Replace the sleep code with your Python function to generate images, text, or run any machine learning workload
    time.sleep(seconds)  
    
    return

if __name__ == '__main__':
    runpod.serverless.start({'handler': handler })
