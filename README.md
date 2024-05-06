<div align="center">
    <img src="./docs//logo.svg" width="200"/>
</div>

# 书生·浦语大模型

> 最新更新：2024年4月25日

> 该项目为记录、整理、分享学习书生·浦语大模型大模型相关知识，如有问题欢迎指出。

# 项目简介

书生·浦语（InternLM）是由上海人工智能实验室与商汤科技联合香港中文大学、复旦大学发布的新一代大语言模型，InternLM 是在过万亿 token 数据上训练的多语千亿参数基座模型。通过多阶段的渐进式训练，InternLM 基座模型具有较高的知识水平，在中英文阅读理解、推理任务等需要较强思维能力的场景下性能优秀，在多种面向人类设计的综合性考试中表现突出。在此基础上，通过高质量的人类标注对话数据结合 RLHF 等技术，使得 InternLM 可以在与人类对话时响应复杂指令，并且表现出符合人类道德与价值观的回复。

官网地址：[点击传送](https://internlm.intern-ai.org.cn/)

开源地址：[点击传送](https://github.com/InternLM/InternLM)

HuggingFace：[点击传送](https://huggingface.co/internlm)

ModelScope：[点击传送](https://modelscope.cn/organization/Shanghai_AI_Laboratory)

飞书：[点击传送](https://aicarrier.feishu.cn/wiki/RPyhwV7GxiSyv7k1M5Mc9nrRnbd)

OpenXLab 部署教程：[点击传送](https://github.com/InternLM/Tutorial/tree/camp2/tools/openxlab-deploy)

# 学习课程
> 以下学习课程围绕官方GitHub仓库进行学习，学习内容如下。后期会更新自己在本地使用LMDeploy部署模型及使用XTuner微调模型的学习笔记，也将会利用LMDeploy提供的服务API，将本地模型结合到Java开发的[ChatMASTER](https://gitee.com/panday94/chat-master)大模型对话中，该项目已整合ChatGPT、文心一言、智谱清言、讯飞星火及月之暗面等接口，欢迎大家star。

- 第1节：书生·浦语大模型全链路开源体系 
[视频地址](https://www.bilibili.com/video/BV1Vx421X72D/)
[笔记地址](Note1)

- 第2节：轻松玩转书生·浦语大模型趣味 Demo 
[项目地址](https://github.com/InternLM/InternLM)
[视频地址](https://www.bilibili.com/video/BV1AH4y1H78d/) 
[文档地址](https://github.com/InternLM/Tutorial/blob/camp2/helloworld/hello_world.md)
[笔记&作业地址](Note2)

- 第3节：“苘香豆":零代码搭建你的 RAG 智能助理
[项目地址](https://github.com/InternLM/HuixiangDou)
[视频地址](https://www.bilibili.com/video/BV1QA4m1F7t4/)
[文档地址](https://github.com/InternLM/Tutorial/blob/camp2/huixiangdou/readme.md)
[笔记&作业地址](Note3)
  - 在[茴香豆 Web 版](https://openxlab.org.cn/apps/detail/tpoisonooo/huixiangdou-web)中创建自己的知识问答助手
  - 本地部署“苘香豆"web版（可选未完成）
  - 在 InternLM Studio 上部署茴香豆技术助手
  - 茴香豆进阶（选做）

- 第4节：XTuner 微调 LLM:1.8B、多模态、Agent
  [项目地址](https://github.com/InternLM/XTuner)
  [视频地址](https://b23.tv/QUhT6ni)
  [文档地址](https://github.com/InternLM/Tutorial/blob/camp2/xtuner/readme.md)
  [笔记&作业地址](Note4)
  - XTuner 微调个人小助手认知
    - 数据集准备
    - 模型准备
    - 配置文件选择
    - 模型训练
    - 模型转换、整合、测试及部署

- 第5节：LMDeploy 量化部署 LLM 实践
  [项目地址](https://github.com/InternLM/LMDeploy)
  [视频地址](https://www.bilibili.com/video/BV1tr421x75B/)
  [文档地址](https://github.com/InternLM/Tutorial/blob/camp2/lmdeploy/README.md)
  [笔记&作业地址](Note5)
  - 环境部署
  - LMDeploy模型对话
  - LMDeploy模型量化(lite)
  - LMDeploy服务(serve)
  - Python代码集成
  - 使用LMDeploy运行视觉多模态大模型llava(拓展)

- 第6节：Lagent & AgentLego 智能体应用搭建
  [Lagent项目地址](https://github.com/InternLM/Lagent)
  [AgentLego项目地址](https://github.com/InternLM/AgentLego)
  [视频地址](https://www.bilibili.com/video/BV1Xt4217728/)
  [文档地址](https://github.com/InternLM/Tutorial/tree/camp2/agent)
  [笔记&作业地址](Note6)
  - 概述
  - Lagent：轻量级智能体框架
  - AgentLego：组装智能体“乐高”
  - Agent 工具能力微调

- 第7节：OpenCompass大模型评测实战
  [项目地址](https://github.com/open-compass/opencompass)
  [视频地址](https://www.bilibili.com/video/BV1Pm41127jU/)
  [文档地址](https://github.com/InternLM/Tutorial/blob/camp2/opencompass/readme.md)
  [笔记&作业地址](Note7)

- 第8节：大模型微调数据构造 [视频地址]()
- 第9节：平台工具类补充课程 [视频地址]()

# 优秀项目学习
- 第二期书生浦语大模型实战营第三节笔记整理【茴香豆与RAG 智能助理】【助教视角】 - 知乎
https://zhuanlan.zhihu.com/p/691906701?

- 第二期书生浦语大模型实战营第三节笔记【茴香豆与RAG 智能助理】 - 知乎
https://zhuanlan.zhihu.com/p/692030192?

- [HoK][RAG]基于茴香豆Web知识库的王者荣耀英雄故事助手[书生·浦语大模型实战营第二期第三课作业] - 知乎
https://zhuanlan.zhihu.com/p/691827925

- 书生·浦语大模型实战营第二期资源汇总【助教视角】 - 知乎
https://zhuanlan.zhihu.com/p/691047729?

- 使用XTuner微调llama3
https://github.com/SmartFlowAI/Llama3-XTuner-CN/
