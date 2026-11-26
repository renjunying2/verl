# 模型评测

Last updated: 07/14/2026.

不同模型步骤一致，仅以Qwen3-30B为例列举

我们通过 AISBench 评估模型，该工具支持vllm/sglang多种推理后端的评估

## 1.安装方法

~~~bash
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark
pip install -e .
~~~


## 2.下载评估数据集

~~~bash
cd path/to/benchmark/ais_bench/datasets
wget https://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip
unzip math.zip
rm math.zip
~~~

## 3.权重转换

当前verl已经支持mbridge直接保存hf格式模型权重，无需转换即可使用。

如果模型权重不是hf格式，需要先转换为hf格式，再进行评估。

此处参照verl原生[转换方法](../../../../../docs/advance/checkpoint.rst)

## 4.vllm推理评测

**启动vllm_server服务**

通过以下命令拉起推理服务端，需要修改的参数：model和tensor-parallel-size。

model：保存训练后权重转换完的huggingface模型地址；

tensor-parallel-size：张量并行副本数，TP建议和训练时infer的配置保持一致；

data-parallel-size：数据并行副本数，DP建议和训练时infer的配置保持一致，默认为1；

port：可任意设置空闲端口；

~~~bash
vllm serve /path/to/Qwen3-30B/ \
       --served-model-name auto \
       --gpu-memory-utilization 0.9 \
       --max-num-seqs 24 \
       --max-model-len 22528 \
       --max-num-batched-tokens 22528 \
       --enforce-eager \
       --trust-remote-code \
       --distributed-executor-backend=mp \
       --tensor-parallel-size 8 \
       --data-parallel-size 1 \
       --generation-config vllm \
       --port 8080
~~~

**修改AISBench推理配置启动vllm_client评测**

打开推理配置文件 ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py

host_port需与服务端的port一致，根据模型配置修改model（设为auto）、path（填写模型路径）和max_out_len。
~~~python
from ais_bench.benchmark.models import VLLMCustomAPIChatStream
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="",              # 模型/tokenizer 目录，替换为实际路径
        model="",             # 与 --served-model-name 一致
        request_rate = 0,     # 请求发送速率，0 表示不限制
        retry = 2,            # 请求失败重试次数
        host_ip = "localhost",# 推理服务 IP
        host_port = 8080,     # 与服务端 --port 保持一致
        max_out_len = 512,    # 单次输出最大 token 数
        batch_size=1,         # 并发请求数
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 0.5,      # 采样温度
            top_k = 10,             # top-k 采样
            top_p = 0.95,           # top-p 采样
            seed = None,            # 随机种子，None 表示不固定
            repetition_penalty = 1.03,  # 重复惩罚系数
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
~~~

另起一个窗口进行评测，开启评测命令：
~~~bash
    ais_bench --models vllm_api_stream_chat --datasets math500_gen_0_shot_cot_chat_prompt
~~~
## 5.sglang推理评测
参照 [昇腾 SGLang 最佳实践](../../model_support/examples/ascend_sglang_best_practices.rst)中评测进行