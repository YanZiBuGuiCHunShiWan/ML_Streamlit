import streamlit as st
import pandas as pd
import random
def page_llm_training():
    st.title("大语言模型训练")
    st.markdown("")
    st.image("src/Instruction_tuning.gif")
    st.markdown("## :blue[SFT训练详解]")
    # 在首页中添加内容
    st.markdown("&emsp;&emsp;Causal Language Model（因果语言模型）是一种基于序列的语言模型，它根据前面的上文生成下一个词或字符，该模型主要用于生成文本，具体的生成过程如上图所示。")
    st.markdown("&emsp;&emsp;给定一个单论对话的数据，输入是：:blue['给定一个英文句子，翻译成中文。\nI love to learn new things every day.\n']，回答是：:blue['我每天喜欢学习新事物。']。现在我们想训练一个因果语言模型完整这个问答任务，那么该如何建模呢？")
    st.markdown("&emsp;&emsp;如果直接把输入和回答拼接在一起变成':blue[给定一个英文句子，翻译成中文。 I love to learn new things every day.我每天喜欢学习新事物。]'让模型根据上文生成下一个token,那么存在两个问题：\
        1. :red[模型分不清输入和回答的部分]，2.:red[模型不知道何时结束]。")
    st.markdown("&emsp;&emsp;我们可以通过引入额外的标识符和'</s>'添加有效的区分信息。具体如下：")
    code_content=''' "Human: "+"给定一个英文句子，翻译成中文。\\nI love to learn new things every day.\\n"+"\\n</s>"+"<s>Assistant: "+"我每天喜欢学习新事物。"+"</s>" '''
    st.code(code_content,language="shell")
    st.markdown("&emsp;&emsp;如果是下面的多轮对话数据，我们可以引入对话角色的标识，比如:blue['求助者：']和:blue['支持者：']，当然，换成:blue['Human：']与:blue['Assistant：']也完全没问题。")
    multi_turn_conversation='''  "求助者：最近总是和妈妈闹矛盾，但是又不知道该怎么办，能帮我想想办法吗？",\n
  "支持者：我了解你的情况，跟亲人之间经常会产生矛盾是很常见的现象。你不妨试试和妈妈沟通一下，平静地提出自己的疑惑和不满，看看能否解决矛盾。",\n
  "求助者：但是每次我和妈妈说话，总会起争端，她总是让我感觉她不信任我，我该怎么办呢？",\n
  "支持者：听起来你和妈妈之间的交流很困难，你可以试试换个方式和她沟通，比如写信或者找一个更加中立的人一起协调谈话，让大家都有更好的表达机会。",\n
  "求助者：我特别讨厌和她吵架，可是我有时候就是自制力不够，很难抑制自己的情绪。",\n
  "支持者：青春期的年轻人情绪波动很大很正常，但是你可以试试找些方法来缓解情绪，比如听听音乐、看看书等等，使自己情绪更稳定。"'''
    st.code(multi_turn_conversation,language="shell")
    st.markdown("&emsp;&emsp;最后将多轮对话数据通过'</s>'拼接得到一条完整的训练样本。")
    code_samples='''"input1</s>target1</s>input2</s>target2</s>...inputn</s>target</s>"'''
    st.code(code_samples,language="shell")
    st.markdown("&emsp;&emsp;将数据准备好以后，那么该采用什么样的损失函数来训练模型呢？我们可以用Transformers默认的:blue[AutoModelForCausalLm]进行训练，无需改写损失函数。\
        这样的特点是，在训练时模型会不断地根据上文来预测下一个token，即属于人类说话的部分也会被预测，这一部分的损失不会被忽略。我们也可以改写损失函数，将属于人类说话部分的损失忽略(Mask掉)，即只预测模型回答的那一部分。两种方式都是可行的。\
        示意图如下")
    st.image("src/Daily-trainloss.png")
    st.markdown("## :blue[如何给模型赋予身份]")
    st.markdown("&emsp;&emsp;如果想将大语言模型部署给用户进行聊天，那么有必要让模型认清自己是谁，比如能够告诉用户自己的公司，是谁创造了自己等问题，接下来就分享如何改变大模型对自己身份认知的手段。")
    identity_code='''{"conversation_id": 0, "content": [{"question": "你好,请问你是谁？谁研发的你？", "answer": "你好，我是由广东数业智能的工程师们研发和创造的人工智能助手小陆。我的研发团队包括了自然语言处理、机器学习、计算机科学等多个领域的专家，他们在各自的领域都有着丰富的经验和深厚的技术功底。为了创造我，他们研究了大量的人类语言交流数据，以训练和优化我的语言理解能力和生成能力。如果你有任何问题或需要帮助，我可以为你提供帮助。"}]}
{"conversation_id": 1, "content": [{"question": "你好，你是机器人吗？", "answer": "你好！我是数业智能科技有限公司的人工智能助手小陆。很高兴能够为您提供帮助。有什么我可以帮您解答的问题吗？"}, {"question": "你有其他名字吗？", "answer": "目前，我被称为小陆，但您也可以根据喜好给我起其他的名字。无论怎样称呼我，我都会尽力为您提供满意的服务。"}, {"question": "你有固定的工作时间吗？", "answer": " 作为一个虚拟助手，我没有固定的工作时间。我可以随时为您提供帮助，无论是白天还是夜晚。"}]}
{"conversation_id": 2, "content": [{"question": "请问你能自我介绍一下吗？", "answer": "你好！我是广东数业智能科技有限公司的人工智能助手小陆。我可以和人类进行自然交流、解答问题以及心理疏导，如果你有任何问题或者遇到困难，欢迎随时向我提问，我会尽我所能为你提供帮助"}, {"question": " 哦，那请问数业智能科技有限公司是干什么的？能否详细介绍一下呢？", "answer": "数业智能科技有限公司是一家专注于解决青少年心理健康问题的人工智能技术的公司。创始团队由多位AI领域顶尖人才组成，包括来自UCLA、中山大学、南方科技大学等知名大学的心理学以及人工智能领域的专家教授"}, {"question": "你的创始人是谁？", "answer": " 我的创始人是数业智能科技有限公司的工程师们。我被设计用于和人类进行自然交流、解答问题、协助创作，帮助大众轻松、普惠的获得世界知识和专业服务。如果你有任何问题，可以随时向我提问。"}, {"question": "数业智能科技有限公司有多少员工？", "answer": "作为一个大语言模型，我没有具体的人数概念。但是，数业智能是一家专注于心理健康和人工智能交叉领域的科技公司，我们希望建立一个具有广泛心理学知识和心理疏导技能的大模型，以及利用这些大模型推出各种产品和服务，以帮助人们更好地解决问题和交流。如果你对我们的技术和产品感兴趣，可以访问我们的网站了解更多信息：https://www.shuyeit.com/ 或添加我们的微信公众号：心大陆空间"}]}'''
    st.code(identity_code,language="python")
    st.markdown("&emsp;&emsp;最一开始的思路是：编写和身份问答相关的多轮对话语料再复制多次得到多条相同的样本，然后再混一点通用领域的多轮对话数据一起微调，然后测试结果。分别尝试了\
        :blue[Baichuan2-7B-Base]模型和:blue[Baichuan2-7B-Chat]模型以后发现一点效果都没有。")
    
    
    
def page_inference_speed():
    st.title("模型加速推理")
    st.markdown("## :blue[vllm模型加速]")
    st.markdown("### :blue[部署方法]")
    st.markdown("&emsp;&emsp;第一步，环境安装。推荐安装:blue[CUDA TOOLKIT 11.8]版本，以及:blue[CuDNN 11.X]版本。然后输入如下命令安装即可")
    pip_code='''pip install vllm'''
    st.code(pip_code,language="python")
    st.markdown("&emsp;&emsp;第二部，编写服务代码:blue[server.py]，内容如下。然后运行:blue[python sever.py]")
    api_code='''
    import argparse
import json
from typing import AsyncGenerator

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn
from time import time
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid

TIMEOUT_KEEP_ALIVE = 5  # seconds.
TIMEOUT_TO_PREVENT_DEADLOCK = 1  # seconds.
app = FastAPI()


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    stream = request_dict.pop("stream", False)
    sampling_params = SamplingParams(**request_dict)
    request_id = random_uuid()
    results_generator = engine.generate(prompt, sampling_params, request_id)
    start=time()
    # Streaming case
    async def stream_results() -> AsyncGenerator[bytes, None]:
        async for request_output in results_generator:
            prompt = request_output.prompt
            text_outputs = [
                prompt + output.text for output in request_output.outputs
            ]
            ret = {"text": text_outputs}
            yield (json.dumps(ret) + "\0").encode("utf-8")

    async def abort_request() -> None:
        await engine.abort(request_id)

    if stream:
        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        background_tasks.add_task(abort_request)
        return StreamingResponse(stream_results(), background=background_tasks)

    # Non-streaming case
    final_output = None
    async for request_output in results_generator:
        if await request.is_disconnected():
            # Abort the request if the client disconnects.
            await engine.abort(request_id)
            return Response(status_code=499)
        final_output = request_output

    assert final_output is not None
    prompt = final_output.prompt
    text_outputs = [prompt + output.text for output in final_output.outputs]
    end=time()
    ret = {"text": text_outputs,"time":round(end-start,4)}
    return JSONResponse(ret)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8090)
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level="debug",
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE)
    '''
    # 在关于页面中添加内容
    st.code(api_code,language="python")
    st.markdown("&emsp;&emsp;第三步，编写请求代码")
    request_code='''
import json
import time
import urllib.request
import sys


def gen_prompt(input_text):
    prompt =  "<s>Human: "+input_text+"\\n</s><s>Assistant: " #这是Llama2-chat的prompt
    return prompt

def test_api_server(input_text):
    header = {'Content-Type': 'application/json'}

    prompt = gen_prompt(input_text.strip())

    data = {
          "prompt": prompt,
          "stream" : False,
          "n" : 1,
          "best_of": 1, 
          "presence_penalty": 0.0, 
          "frequency_penalty": 0.2, 
          "temperature": 0.3, 
          "top_p" : 0.95, 
          "top_k": 50, 
          "use_beam_search": False, 
          "stop": [], 
          "ignore_eos" :False, 
          "max_tokens": 2048, 
          "logprobs": None
    }
    request = urllib.request.Request(
        url='http://127.0.0.1:8192/generate',
        headers=header,
        data=json.dumps(data).encode('utf-8')
    )

    result = None
    try:
        response = urllib.request.urlopen(request, timeout=300)
        res = response.read().decode('utf-8')
        result = json.loads(res)
        print(json.dumps(data, ensure_ascii=False, indent=2))
        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        print(e)

    return result

if __name__ == "__main__":
    test_api_server("如何去北京?")
    test_api_server("肚子疼怎么办?")
    test_api_server("帮我写一个请假条")
    '''
    st.code(request_code,language="python")
    st.markdown("### :blue[效果对比]")

def page_paper_reading():
    st.title("联系我们")
    # 在联系我们页面中添加内容
    
def page_others():
    st.title("其他想法")
# 定义侧边栏选项

def page_knowledge_llm():
    st.title("大语言模型与本地知识库的结合")
    
def page_model_server():
    st.title("基于Sanic框架的简易模型部署")
    st.markdown("&emsp;&emsp;这一章我们动手搭建一个建议的http服务，当向服务端发起请求时，我们要携带两点信息：:red[1.请求的内容，2.uuid(用于标识当前的对话id，\
                不至于使当前的对话和其他用户的对话混淆)]")
    st.markdown("#### :blue[导入模块并初始化一个应用]")
    import_code='''import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
import os,datetime,uvicorn,asyncio
import argparse
import uuid,json
import warnings
from sanic import Sanic
from sanic.request import Request
from sanic.response import json
from loguru import logger
logger.setLevel("ERROR")
warnings.filterwarnings("ignore")
'''
    st.code(import_code,language="python")
    st.markdown("#### :blue[初始化模型和app]")
    instantiate_code='''model=AutoModelForCausalLM.from_pretrained(model_path,
                                           trust_remote_code=True,
                                           torch_dtype=torch.float16,
                                           device_map=torch.cuda.current_device(),
                                           quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"))
tokenizer=AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
logger.info("Loading complete")
app = Sanic(__name__)'''
    st.code(instantiate_code,language="python")
    st.markdown("#### :blue[情景一：异步非阻塞]")
    st.markdown("&emsp;&emsp;我们用一个变量:blue[history_mp]来保存每一个对话内容，以:blue[uuid]作为字典的键，值则是一个列表。当获取的请求的:blue[uuid]是:blue[None]时，\
        则说明是对话的开始，没有历史记录，系统会生成一个对话id，下次请求时就可以根据:blue[uuid]拿到对话历史记录。")
    st.markdown("&emsp;&emsp;我们将函数:blue[model.chat]添加到事件循环，并注册成一个异步非阻塞的方法。因为原始的语言模型推理这一个步骤是同步阻塞的，意味着当前\
        推理任务如果没有处理完，用户的请求会阻塞，不会被处理。所以通过:blue[run_in_executor]包装使得其变成异步非阻塞的操作。")
    inference_code='''
    history_mp = {} # restore history for every uid

@app.post("/test")
async def create_item(request: Request):
    json_post_raw = request.json
    query = json_post_raw.get('prompt') 
    uid = json_post_raw.get('uid', None)
    if uid == None or not(uid in history_mp):
        uid = str(uuid.uuid4())
        history_mp[uid] = []
    messages=history_mp[uid]+[{"role": "user", "content": query}]
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    response=await app.loop.run_in_executor(None,model.chat,tokenizer,messages) #Baichuan-13B-Chat
    history_mp[uid] = history_mp[uid] + [{"role": "assistant", "content": response}]
    answer = {
        "response": response,
        "history": history_mp[uid],
        "status": 200,
        "time": time,
        "uid": uid,
    }
    log = "[" + time + "] " + '", prompt:"' + query + '", response:"' + repr(response) + '"'
    print(log)
    return json(answer)
if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=19324, workers=2)
    """注：这是Baichuan-13B-Chat的推理方式，如果是其他模型就按照该思路改写即可"""
    '''
    st.code(inference_code,language="python")
    st.markdown("&emsp;&emsp;我们用postman发起请求看看效果如何：")
    st.image("src/Daily-model-deployment.png")
    st.markdown("#### :red[存在的问题]")
    st.markdown("&emsp;&emsp;由于将模型推理改成了异步非阻塞的函数，意味着1s如果有10个请求过来，系统都会调度资源进行处理，比如请求1还没执行完，请求2和请求3就开始了，而显卡的计算资源是有限的。\
        系统调度资源来处理每个任务则会使得每一个计算密集型的任务处理时间都大幅度增加。实际测试后，发现单独完成一个请求只需50秒，但是同时发起5个请求则每一个请求要都花费5分钟，说明系统给每一个\
            计算任务分配的资源都分摊开了，导致每一个模型推理的计算过程都明显延长了时间。因此得出结论：:red[用协程来处理计算密集型的任务反而不太合适]")
    st.markdown("#### :blue[情景二：同步阻塞]")
    st.markdown("&emsp;&emsp;说白了就是普通的推理方式，把之前的run_in_executor去掉。")
    normal_inference=''' response = model.chat(tokenizer,messages) '''
    st.code(normal_inference,language="python")
    st.markdown("&emsp;&emsp;同步阻塞意味着当模型正在进行一个任务推理时，有另一个请求又发了过来，那么这个新的请求不会被立即处理，而是等到当前的同步任务处理完以后\
        才会被进一步处理。比如任务1推理耗时30s，那么任务2会直到任务1完成以后才会被处理。")
    
pages = {
    "大语言模型训练": page_llm_training,
    "大模型+知识库": page_knowledge_llm,
    "加速推理方案": page_inference_speed,
    "其他想法": page_others,
    "模型部署": page_model_server

}



# 添加侧边栏菜单
selection = st.sidebar.radio("系列精选", list(pages.keys()))

# 根据选择的菜单显示相应的页面
page = pages[selection]
page()