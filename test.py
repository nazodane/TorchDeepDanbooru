from PIL import Image
import numpy as np
import torch
import tqdm

import deep_danbooru_model

model = deep_danbooru_model.DeepDanbooruModel()
model.load_state_dict(torch.load('model-resnet_custom_v3.pt'))

model.eval()
model.half()
model.cuda()

pic = Image.open("test.jpg").convert("RGB").resize((512, 512))
a = np.expand_dims(np.array(pic, dtype=np.float16), 0) / 255


with torch.no_grad():# , torch.autocast("cuda")
    x = torch.from_numpy(a).cuda()

    # non-jit first run
    y = model(x)[0].detach().cpu().numpy()

    print("non-jit")
    # non-jit measure performance -> 79.18it/s on RTX 3060
    for n in tqdm.tqdm(range(10)):
        model(x)

    # jit script first run
    # pip install monkeytype -> error, don't install
    scripted_fn = torch.jit.script(model, example_inputs=[x])
#    torch._C._jit_pass_inline(scripted_fn.graph) 
    z = scripted_fn(x)[0].detach().cpu().numpy()

    print("script")
    # jit script measure performance -> 6.88it/s on RTX 3060
    for n in tqdm.tqdm(range(10)):
        scripted_fn(x)

    # jit traced first run
    traced_fn = torch.jit.trace(model, example_inputs=[x])
#    torch._C._jit_pass_inline(traced_fn.graph) 
    aa = traced_fn(x)[0].detach().cpu().numpy()

    print("traced")
    # jit traced measure performance -> 94.39it/s on RTX 3060
    for n in tqdm.tqdm(range(10)):
        traced_fn(x)

    # torchdynamo+triton first run
    dynamo_fn = torch.compile(model, fullgraph=True)
    za = dynamo_fn(x)[0].detach().cpu().numpy()

    print("compiled - triton")
    # torchdynamo+triton measure performance -> 105.58it/s on RTX 3060
    for n in tqdm.tqdm(range(10)):
        dynamo_fn(x)

    torch._dynamo.reset()
    # torchdynamo+onnxrt first run
    # wrong: https://github.com/pytorch/pytorch/blob/e737a8486f2b38c699d6890be295dd6e6d6417ea/docs/source/compile/get-started.rst
    #        https://github.com/pytorch/pytorch/blob/6b1d6750b98ee1e3ad2fb715194b4a5b247b782d/torch/_dynamo/backends/tensorrt.py
    # pip install onnxruntime-gpu
    # right: https://github.com/pytorch/TensorRT/blob/cae6b7c694db1b6a5f287f9fc4e6c40ba08e1079/py/torch_tensorrt/dynamo/backend/backends.py#L26
    # pip install torch-tensorrt
    import torch_tensorrt
    dynamot_fn = torch.compile(model, fullgraph=True, backend="torch_tensorrt") # not "tensorrt"
    zb = dynamot_fn(x)[0].detach().cpu().numpy()

    print("compiled - tensorrt")
    # torchdynamo+tensorrt measure performance -> 67.36it/s on RTX 3060
    for n in tqdm.tqdm(range(10)):
        dynamot_fn(x)

#    torch.onnx.export(scripted_fn, [x],  "scripted.onnx")
##    torch.onnx.export(traced_fn, [x],  "traced.onnx")

# TensorRT model generating and using on Ubuntu 22.04
# $ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
# $ sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
# $ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
# $ sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /"
# $ sudo apt-get update
# $ sudo apt install cuda-11-8
#
# $ python -m pip install colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com
# $ ~/.local/bin/polygraphy surgeon sanitize scripted.onnx --fold-constants --output scripted_folded.onnx
# $ sudo apt install libnvinfer-bin
# $ /usr/src/tensorrt/bin/trtexec --onnx=scripted_folded.onnx --saveEngine=scripted_folded.trt
#
# $ sudo apt install python3-libnvinfer # don't install pip's nvidia-tensorrt
#
# $ export CPATH=$CPATH:/usr/local/cuda/include
# $ export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda/lib64
# $ pip install pycuda # don't install apt's python3-pycuda

#import os
#if not os.path.exists("scripted_folded.trt"):
#    exit()

#import tensorrt as trt
#import pycuda.autoinit
#import pycuda.driver as cuda
#def load_engine(trt_runtime, engine_path):
#    with open(engine_path, 'rb') as f:
#        engine_data = f.read()
#    engine = trt_runtime.deserialize_cuda_engine(engine_data)
#    return engine
#
#TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
#trt_runtime = trt.Runtime(TRT_LOGGER)
#engine = load_engine(trt_runtime, "scripted_folded.trt")
#stream = cuda.Stream()
#ctx = engine.create_execution_context()


def trt_fn(input):
    r = np.empty(9176, dtype = np.float16)

    d_input = cuda.mem_alloc(1 * 3 * 512 * 512 * 2) # half float
    d_output = cuda.mem_alloc(1 * 9176 * 2)

    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    cuda.memcpy_htod_async(d_input, input, stream)
    ctx.execute_async_v2(bindings, stream.handle, None)
    cuda.memcpy_dtoh_async(r, d_output, stream)
    stream.synchronize()
    return r

# trt first run
#ab = trt_fn(a)

# trt measure performance -> 56.91it/s on RTX 3060
#print("trt")
#for n in tqdm.tqdm(range(10)):
#    trt_fn(a)

for i, p in enumerate(y):
    if p >= 0.5:
        print("y:", model.tags[i], p)
    if z[i] >= 0.5:
        print("z:", model.tags[i], z[i])
    if aa[i] >= 0.5:
        print("aa:", model.tags[i], aa[i])
    if za[i] >= 0.5:
        print("za:", model.tags[i], za[i])
    if zb[i] >= 0.5:
        print("zb:", model.tags[i], zb[i])
#    if ab[i] >= 0.5:
#        print("ab:", model.tags[i], ab[i])
