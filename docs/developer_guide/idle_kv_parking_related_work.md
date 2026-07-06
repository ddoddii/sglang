# Idle KV Parking — 아이디어 정리, 관련 연구, 현실성 평가

> 참고 논문: **OrbitCache: Pushing the Limits of In-Network Caching for Key-Value Stores** (Gyuyeong Kim, NSDI'25)
> 이 문서는 "KV cache를 P node에만 두지 말고, 네트워크로 연결된 여러 GPU 서버에 KV를 돌아다니게 하다가 필요할 때 부른다 — 재계산보다 빠르니까"라는 아이디어를 OrbitCache에 비추어 다듬고, 관련 연구를 정리하고, 현실성을 평가한다.

---

## 1. OrbitCache가 실제로 하는 것 (정확히)

- **문제**: 프로그래머블 스위치(Intel Tofino)의 in-network cache는 스위치 SRAM에 hot item을 저장하는데, 하드웨어 제약(key ≤16B, value ≤128B)으로 **작은 아이템만** 캐시 가능. 실제 워크로드(수십~1024B)를 못 담음.
- **아이디어**: hot item을 스위치 메모리에 **저장**하지 않고, **cache packet 형태로 스위치 data plane을 계속 recirculation(내부 loopback 포트로 재순환)** 시킨다. 즉 데이터를 "정지 저장"이 아니라 "계속 순환하는 상태"로 유지.
- **요청 처리**: 요청은 recirculation 하지 않고 스위치 메모리에 작은 request metadata만 유지. 순환 중인 cache packet이 지나갈 때 PRE(packet replication engine)로 **clone**해서 대기 중인 여러 요청에 동시에 응답.
- **핵심 제약 관리**: recirculation 포트는 파이프라인당 1개뿐이라 병목. 그래서 **요청은 절대 순환시키지 않고, 소수의 상수 개수 cache packet만** 순환시켜 포트 대역폭을 아낌.
- **coherence**: invalidation 기반 프로토콜(읽기/쓰기 혼재하는 KV store이므로 필요).

**한 줄 요약**: "저장 용량이 부족한 매체(스위치 SRAM)에서는, 데이터를 계속 흐르게 두는 것이 저장하는 것보다 낫다 — 단, 아이템이 작고 개수가 적을 때만."

---

## 2. LLM KV cache로 유추(analogy) — 무엇이 옮겨지고 무엇이 깨지는가

사용자 아이디어: *네트워크가 빠르니 KV를 노드마다 돌아다니게 하다가 필요할 때 부른다. 재계산보다 빠르니까.*

### 2.1 옮겨지는(valid) 부분

| OrbitCache 통찰 | LLM KV cache 대응 | 성립? |
|---|---|---|
| "저장 제약을 데이터 이동으로 우회" | 단일 노드 HBM 용량 한계를 **풀링된 원격 메모리 + 빠른 인터커넥트**로 우회 | ✅ 성립 (이미 연구 주류) |
| "인터커넥트가 빠르면 이동이 유리" | RDMA/NVLink fetch가 **재-prefill(recompute)보다 빠름** | ✅ 성립 (아래 3.3 정량 평가) |
| "순환 packet을 clone해 다수 요청 처리" | 하나의 캐시된 prefix를 **여러 요청이 공유**(prefix sharing) | ⚠️ 부분 — 이미 RadixAttention/prefix cache가 함 |
| coherence 프로토콜 | KV는 **append-only/immutable prefix** → coherence가 오히려 **더 단순** | ✅ 유리 (LLM 쪽이 쉬움) |

### 2.2 깨지는(invalid) 부분 — 여기가 중요

1. **데이터 크기 스케일 불일치**
   OrbitCache 아이템 ≤1024B(단일 패킷). LLM KV는 Llama-3.1-8B 기준 **128 KB/token**. 대화 prefix 4000 토큰 = **~512 MB**. 이건 "패킷"이 아니라 대용량 텐서다.

2. **"계속 순환(in-flight)"은 KV에서 이득이 없다**
   OrbitCache에서 순환은 스위치 SRAM 용량 한계를 **벗어나게** 해준다(순환 중엔 SRAM을 안 씀). 하지만 512MB KV를 네트워크에 **상시 in-flight로 돌리면 대역폭을 영구 점유**할 뿐, 어떤 용량도 절약되지 않는다(어딘가의 HBM/DRAM에서 나가고 들어오는 버퍼는 계속 필요). 즉 **"literal 순환"은 LLM KV에 부적합**.
   → 올바른 정식화는 **"순환(in motion)"이 아니라 "분산 풀에 at-rest로 두고 필요할 때 on-demand fetch/prefetch"**.

3. **스위치 in-network 처리 불가**
   512MB KV를 Tofino data plane에서 clone/처리하는 것은 물리적으로 불가능. "in-network"는 이 맥락에서 **RDMA 네트워크를 통한 원격 메모리 풀링**으로 재해석해야지, 스위치 ASIC 처리로 해석하면 안 된다.

### 2.3 다듬어진 아이디어 (제안 정식화)

> **"KV를 특정 P node에 고정 저장하지 않고, 클러스터의 유휴 GPU HBM + CPU DRAM + (원격 노드)를 빠른 인터커넥트로 묶은 분산 KV 풀에 배치한다. Tool-call 유휴시간에 KV를 유휴 자원으로 이동(park)시켜 두고, 다음 턴에 필요한 노드로 prefetch/fetch한다. 대용량 prefix는 재계산보다 fetch가 빠르기 때문이다."**

- "돌아다니게 한다"의 실체 = **at-rest 분산 배치 + idle-time 재배치(migration) + on-demand prefetch**. (상시 in-flight ❌)
- 본 저장소의 `idle_kv_parking_design.md`(Design A: 유휴 P GPU radix로 park, 초과 시 CPU DRAM 강등)가 이 정식화의 구체적 1차 구현이다.

---

## 3. 관련 연구

### 3.1 분산 KV 풀 / 메모리 disaggregation (아이디어의 주류 계보)

| 연구 | 핵심 | 관계 |
|---|---|---|
| **Mooncake** (FAST'25 Best Paper) | KVCache-centric 아키텍처. 클러스터의 유휴 CPU/DRAM/SSD/RDMA를 묶어 disaggregated KVCache 풀 구성. Kimi 프로덕션. | "돌아다니는 KV 풀"의 가장 근접한 실체. **직접 경쟁/기반** |
| **MemServe** (2024) | Elastic memory pool(MemPool)로 GPU HBM+CPU DRAM 전체를 관리. prompt-tree locality-aware 스케줄링으로 재사용 극대화 | 풀링+locality 라우팅. 본 아이디어의 상위 프레임워크 |
| **LMCache** | GPU/CPU/disk/원격에 KV 저장·재사용, 인스턴스 간 공유. RDMA CPU-driven access | 프로덕션 KV 레이어 |
| **KVDirect** (2025) | Distributed disaggregated inference, GPU-native RDMA로 KV 전송 | 전송 계층 |
| **Unified KV Pooling** (2025/26) | long-context용 통합 KV 풀 | 최신 풀링 |

### 3.2 재계산 vs. fetch 트레이드오프 (아이디어의 핵심 전제 검증 연구)

| 연구 | 핵심 |
|---|---|
| **CacheGen** (SIGCOMM'24) | KV를 압축·스트리밍. **명시적으로 "큰 KV는 전송이, 작은 KV/저대역폭은 재계산이 유리"**라고 규정. 대역폭 낮아지면 압축률↑ 또는 recompute로 전환. KV 3.5–4.3× 압축, fetch+처리 지연 3.2–3.7× 감소 |
| **Cake / "Compute or Load KV Cache? Why not both?"** | prefill 재계산과 prefix load를 **양방향 동시** 수행(앞→뒤 compute + 뒤→앞 load)해 둘 중 빠른 쪽으로 수렴 |
| **Asynchronous KV Cache Prefetching** | 계산과 KV 로드 overlap하는 비동기 prefetch | 

→ 사용자의 "fetch가 recompute보다 빠르다"는 전제는 **이미 문헌이 조건부로 검증**함(큰 prefix + 충분한 대역폭에서 참).

### 3.3 Multi-turn / agentic 재사용 (사용자 워크로드에 가장 근접)

| 연구 | 핵심 | 관계 |
|---|---|---|
| **AttentionStore / CachedAttention** (ATC'24) | 세션 비활성 시 KV를 계층적 저장소(DRAM/SSD)에 보관, 재활성 시 fetch해 **new token만 partial prefill**. layer-wise 선로딩 + 비동기 저장으로 계산과 overlap | **본 아이디어의 multi-turn 버전 원형.** "tool-call 유휴시간 park"와 거의 동형(단 원격 P GPU가 아니라 로컬 계층 저장소) |
| **"Not All Prefills Are Equal: PPD Disaggregation for Multi-turn"** | multi-turn 전용 PPD disaggregation | experiments의 vllm-ppd 계열과 직접 연관 |
| **KVCOMM** | multi-agent 시스템 간 cross-context KV 통신·재사용 | agentic 확장 |

### 3.4 CXL 기반 (하드웨어 대안 경로)

- **TraCT** (CXL 공유 메모리를 KV 전송 substrate + rack-wide prefix cache로), **CXL-SpecKV**, **SAC**(sparse attention + CXL), **Exploring CXL-based KV Cache Storage** (NeurIPS'24 MLForSys).
- → "원격 풀"을 네트워크(RDMA) 대신 CXL로 구현하는 갈래. 본 아이디어의 하드웨어 대체재.

### 3.5 In-network(스위치) 계열 — literal 유추의 위치

- OrbitCache/NetCache/DistCache 등은 **KV *store*(작은 아이템)** 용이며, **LLM KV *cache*에 스위치 in-network를 적용한 연구는 사실상 없음**.
- 이는 (a) 진짜 gap이지만 동시에 (b) 2.2에서 본 대로 **스케일상 비현실적**. 따라서 "스위치에서 KV를 순환시킨다"는 방향은 연구 공백이 아니라 **부적합 영역**으로 보는 것이 맞다.

---

## 4. 현실성 평가

### 4.1 핵심 전제 정량 검증 (Llama-3.1-8B, A6000 기준)

- KV 크기: **128 KB/token** → 2000 tok ≈ 262 MB, 4000 tok ≈ 524 MB.
- **재계산(prefill) 비용**: 4000 토큰 prefill ≈ 수백 ms~1s급 (index.md의 SGLang 2P2D TTFT avg 0.825s와 정합).
- **fetch(512MB) 비용**:
  | 경로 | 대역폭 | 512MB fetch |
  |---|---|---|
  | NVLink pair | ~112 GB/s | **~5 ms** |
  | PCIe4 x16 | ~32 GB/s | **~16 ms** |
  | 200Gb IB (RDMA) | ~25 GB/s | ~20 ms |
  | 100 GbE | ~12.5 GB/s | ~41 ms |

→ **모든 경로에서 fetch(5–41ms) ≪ recompute(수백 ms).** 사용자의 핵심 전제는 이 스케일에서 **명백히 성립**. (단, 아래 조건 하에서.)

### 4.2 성립 조건 / 리스크

1. **용량이 진짜 병목**: 세션당 512MB × 동시 세션 수 → GPU HBM만으로는 금방 고갈. 그래서 **계층화(HBM→DRAM→SSD/원격)** 필수. Design 문서의 Tier3 강등이 이 지점.
2. **"상시 순환" 금지**: 4.1의 이득은 **필요할 때 fetch/prefetch**할 때만. 상시 in-flight로 돌리면 대역폭을 영구 소모해 오히려 손해(2.2 항목 2).
3. **라우팅 affinity**: fetch 이득은 "다음 턴이 park 보유 노드로 라우팅"될 때 극대화. cache-aware routing에 의존(Design 문서 Phase 1은 1P1D로 먼저 증명 권장).
4. **현재 테스트베드는 단일 노드(4×A6000)**: "여러 GPU *서버*가 네트워크로 연결"은 아직 아님 → 실제로는 **intra-node GPU↔GPU(PCIe/NVLink) 풀링**. 진짜 multi-node RDMA는 A6000 워크스테이션 NIC 사양 확인 필요. **다행히 intra-node일수록 fetch가 더 빨라 전제는 더 강해짐**.
5. **coherence는 걱정 없음**: KV는 immutable prefix → OrbitCache식 invalidation 불필요. LLM 쪽이 오히려 단순.
6. **압축 병용 여지**: CacheGen처럼 KV 압축을 병용하면 fetch 비용을 3–4× 더 낮출 수 있음(대역폭 부족 노드 대비).

### 4.3 결론

- **전제("fetch < recompute")**: ✅ 큰 prefix + 빠른 인터커넥트에서 참. 문헌·정량 모두 지지.
- **literal 아이디어("KV를 네트워크에 계속 돌아다니게")**: ❌ 스케일·대역폭상 부적합. **"분산 풀 at-rest + idle-time 재배치 + on-demand prefetch"로 재정식화**해야 함.
- **차별화 지점(novelty)**: 분산 KV 풀 자체는 Mooncake/MemServe/AttentionStore로 이미 붐비는 영역. 본 아이디어가 기여를 가지려면 **"agentic tool-call 유휴시간을 명시적 트리거로 삼아, 유휴 *P node의 GPU radix*로 park해 다음 턴 prefix-hit를 만든다"**는 **좁고 구체적인 각도**에 집중하는 것이 현실적. (AttentionStore는 로컬 계층 저장소, Mooncake는 CPU/SSD 풀 — "유휴 P GPU를 원격 tier로 재활용"은 상대적으로 덜 다뤄진 각도.)
- **권장 다음 스텝**: Design 문서 Phase 0(기존 hicache/decode-offload가 이미 주는 이득 측정) → Phase 1(1P1D로 park 메커니즘 증명). 여기에 본 문서의 4.1 수치를 실측으로 대체해 "fetch vs recompute" break-even을 자신의 테스트베드에서 확정.

---

## 5. 참고문헌 (URL)

- OrbitCache (NSDI'25): https://www.usenix.org/conference/nsdi25/presentation/kim
- Mooncake (FAST'25): https://arxiv.org/abs/2407.00079 · https://github.com/kvcache-ai/Mooncake
- MemServe: https://arxiv.org/abs/2406.17565
- CacheGen (SIGCOMM'24): https://arxiv.org/abs/2310.07240
- CacheBlend: https://arxiv.org/pdf/2405.16444
- AttentionStore / CachedAttention (ATC'24): https://arxiv.org/abs/2403.19708 · https://www.usenix.org/conference/atc24/presentation/gao-bin-cost
- KVDirect: https://arxiv.org/pdf/2501.14743
- "Compute or Load KV Cache? Why not both?" (Cake): https://openreview.net/pdf?id=cK0kUzocJW
- TraCT (CXL): https://www.researchgate.net/publication/398979928
- KVCOMM (multi-agent): https://arxiv.org/pdf/2510.12872
