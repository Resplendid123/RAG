## 1. 产品概述

InferX 是面向企业级 LLM 推理场景的高性能服务平台,支持主流开源大语言模型(包括 Llama、Qwen、DeepSeek、ChatGLM 等系列)的私有化部署与弹性扩缩容。平台的设计目标是:在不损失推理质量的前提下,把单位 GPU 卡的吞吐量提升 3-5 倍,把 P99 端到端延迟控制在 200ms 以内。InferX 通过模型编译、显存复用、连续批处理(Continuous Batching)、推测解码(Speculative Decoding)等核心优化手段,在 7B-70B 参数规模上提供生产可用的推理服务。

InferX 适用于智能客服、内容审核、代码助手、企业知识库问答等多种场景。平台提供标准化的 RESTful API,兼容 OpenAI 接口协议,用户可零成本切换。系统支持 Kubernetes 原生部署,提供 Helm Chart 和 Operator,具备完善的灰度发布、流量切分、滚动升级能力。

## 2. 核心概念

理解 InferX 的架构,需要先掌握以下核心概念。

**Model(模型)**:指托管在平台上的具体模型实例。每个 Model 由 model_id 唯一标识,包含模型权重、tokenizer 配置、推理参数(temperature、top_p 等)。一个 Model 在运行时可能对应多个 Engine。

**Engine(引擎)**:Engine 是真正执行推理的进程,持有 GPU 显存并维护 KVCache 池。每个 Engine 只能加载一个 Model,但可以通过副本(Replica)实现水平扩展。Engine 启动后向 Scheduler 注册心跳,超时未注册的 Engine 会被自动剔除。

**Replica(副本)**:Replica 是 Engine 的逻辑副本,由 Scheduler 管理。用户声明期望副本数(replicas),Scheduler 根据当前负载自动扩缩容。Engine 故障时,Replica 会被自动迁移到健康节点。

**Scheduler(调度器)**:Scheduler 是 InferX 的"大脑",负责接收推理请求、查询可用的 Engine 集合、选择最优 Engine、转发请求并返回结果。Scheduler 内置多种调度策略:LeastLoaded、LatencyAware、PrefixCacheAware 等。默认策略为 LeastLoaded。

**Router(路由层)**:Router 接收外部 HTTP/gRPC 请求,做协议解析、限流、认证,然后转发给 Scheduler。Router 本身无状态,水平扩展友好。

**KVCache 池**:Engine 内部维护的显存缓存,用于复用历史 token 的 Key/Value 张量,避免每次重新计算。PrefixCacheAware 调度策略会优先将相同 prompt 前缀的请求路由到同一 Engine,最大化 KVCache 命中率。

## 3. 系统架构

InferX 采用经典的 Control Plane / Data Plane 分离架构。

Data Plane 由 Router、Scheduler、Engine 三层组成,负责处理在线推理请求。请求路径为:Client → Router → Scheduler → Engine → Scheduler → Router → Client。Engine 之间的通信采用 NCCL 进行张量并行(当模型超过单卡显存时)。

Control Plane 由 Model Registry、Replica Controller、Metrics Collector 组成,负责模型注册、副本管理、监控采集。Model Registry 存储模型元信息(模型路径、tokenizer 路径、推理参数);Replica Controller 监听 Engine 心跳并根据策略扩缩容;Metrics Collector 周期性地从各组件拉取 Prometheus 格式的指标。

用户部署时,Control Plane 组件部署在管理集群,Data Plane 部署在业务集群,两者通过 mTLS 加密通道通信。

## 4. API 参考

InferX 提供 OpenAI 兼容的 RESTful API,主要端点如下。

**POST /v1/chat/completions**:聊天补全端点,接受 messages 数组(角色包括 system、user、assistant),返回模型生成的回复。请求体核心参数:model(必填,模型 ID)、messages(必填,对话历史)、temperature(可选,默认 0.7,范围 0-2)、max_tokens(可选,默认 512)、stream(可选,默认 false,设为 true 时返回 SSE 流)。

**POST /v1/embeddings**:文本向量化端点,接受 input 字符串或字符串数组,返回 1024 维的稠密向量。请求体参数:model(必填)、input(必填,单条或批量)、encoding_format(可选,值为 float 或 base64,默认 float)。

**POST /v1/completions**:原始文本补全端点,接受 prompt 字符串,返回续写结果。适用于非对话场景,参数与 chat/completions 类似但无 messages 结构。

**GET /v1/models**:列出当前可用的所有模型,返回模型 ID 列表及所有者、创建时间等元信息。

**GET /healthz**:健康检查端点,返回 200 表示服务正常,503 表示部分 Engine 不可用但仍可服务部分请求。

**GET /metrics**:Prometheus 指标端点,供监控系统拉取。

所有 API 均需要 Bearer Token 鉴权,Token 通过控制台申请。请求体大小限制为 4MB,响应超时默认为 30s(可配置)。

## 5. 性能调优指南

InferX 提供多种性能调优手段,以下是在生产环境中验证有效的关键策略。

**1. 启用 Continuous Batching(连续批处理)**:传统静态批处理必须等待批次内所有请求完成后才能处理下一批,导致 GPU 利用率不均。InferX 默认开启 Continuous Batching,在 decode 阶段动态插入新请求,可将吞吐量提升 2-3 倍。

**2. 启用 Prefix Caching(前缀缓存)**:对于多轮对话或具有相同 system prompt 的场景,Prefix Caching 命中率通常可达 60-80%,首次 token 延迟(TTFT)可降低 40%。

**3. 调整 gpu_memory_utilization**:该参数(取值 0-1)控制 Engine 占用的 GPU 显存比例。默认 0.85,推荐 0.9 以提升并发,但需预留足够显存给激活值。

**4. 启用 Speculative Decoding(推测解码)**:用小模型(drafter)先猜测 5-10 个 token,大模型一次验证,可将 decode 速度提升 1.5-2 倍,但会增加一定的显存开销。

**5. 调高 max_num_seqs**:该参数控制单 Engine 最大并发序列数。默认 256,根据实际请求长度调整。请求较短时可调高到 512 或 1024。

**6. 使用量化模型**:INT8 量化几乎无精度损失,显存占用减半;INT4 量化(W4A16)显存再降一半,适合吞吐量优先场景。

## 6. 监控与告警

InferX 暴露丰富的 Prometheus 指标,关键指标包括。

- `inferx_request_total`:总请求数,标签包含 model、status(2xx/4xx/5xx)、endpoint。
- `inferx_request_duration_seconds`:请求延迟直方图,标签包含 model、endpoint。P50/P95/P99 是核心 SLO 指标。
- `inferx_tokens_per_second`:每秒生成 token 数,衡量 decode 速度。
- `inferx_time_to_first_token_seconds`:首 token 延迟(TTFT),反映 Prefill 效率。
- `inferx_kv_cache_usage_ratio`:KVCache 使用率,持续高于 0.9 需要扩容。
- `inferx_queue_length`:等待调度的请求队列长度。
- `inferx_engine_replica_count`:当前各 Engine 的副本数。

推荐告警规则:P99 延迟 > 500ms 持续 5 分钟;TTFT > 100ms 持续 5 分钟;KVCache 使用率 > 0.95 持续 2 分钟;5xx 错误率 > 1% 持续 1 分钟。

## 7. 故障排查

**问题 1:Engine 启动失败,日志报 CUDA OOM**
原因通常是 gpu_memory_utilization 过高或模型加载时静态分配超出预期。处理步骤:(1) 通过 `kubectl logs` 查看具体报错堆栈;(2) 将 gpu_memory_utilization 调低到 0.8 重试;(3) 启用 INT8 量化减少显存占用;(4) 若仍失败,使用 tensor_parallel_size 将模型切分到多卡。

**问题 2:P99 延迟突然飙高**
按以下顺序排查:(1) 查看 Engine 副本数是否充足,Replica Controller 可能因节点故障自动缩容;(2) 检查 KVCache 使用率,接近 1.0 时新请求会排队等待;(3) 检查是否有大请求(>8K tokens)挤占资源,可配置 max_model_len 限制;(4) 查看 GPU 温度和 SM 利用率,排除硬件问题。

**问题 3:Tokenizer 不匹配导致输出乱码**
确认 Model Registry 中配置的 tokenizer 路径与模型权重一致。不同版本的 Qwen 模型使用不同 tokenizer,跨版本混用会导致 Unicode 解码异常。

**问题 4:Prefix Caching 命中率低于预期**
检查 system prompt 是否被错误地分散在多个请求中(每条消息的 system 字段都重新计算)。建议将 system prompt 抽取为全局 prompt_template,所有请求复用同一前缀。

## 8. 配额与限制

- 单租户最大 Engine 副本数:1000
- 单请求最大长度(input + output):32K tokens(具体上限由 model 决定)
- 请求体最大尺寸:4MB
- API 调用 QPS 上限:默认 100,可在控制台申请提升到 10000
- 集群最大 GPU 卡数:无硬限制,受调度器性能制约(建议 ≤ 10000 卡)
- 指标保留时长:Prometheus 30 天,长期存储需对接 Thanos

## 9. 术语表

- **TTFT**(Time To First Token):从请求进入到返回首个 token 的耗时,反映 Prefill 阶段效率。
- **Continuous Batching**:连续批处理,在 decode 阶段动态插入新请求的调度策略。
- **Speculative Decoding**:推测解码,用小模型猜测大模型输出的并行解码技术。
- **KVCache**:Transformer 推理时缓存历史 token 的 Key/Value 张量的显存区域。
- **Prefix Caching**:基于相同 prompt 前缀复用 KVCache 的优化。
- **Data Plane / Control Plane**:数据面/控制面分离架构,Data Plane 处理请求,Control Plane 处理管理。
- **Engine Replica**:Engine 的逻辑副本,Scheduler 管理的最小调度单元。