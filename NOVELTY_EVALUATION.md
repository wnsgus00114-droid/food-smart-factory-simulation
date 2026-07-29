# FlowTwin-Guard 신규성·평가·출판 게이트

> 문서 기준일: 2026-07-27
>
> 대상 구현: FlowTwin-Guard 연산자 `0.1.0`, `flowtwin_benchmark_contract.json`/runner `0.2.0`
>
> 연구 단계: 합성 HTST 사전검증 및 논문 설계. 현장·제품안전·HACCP 검증이 아니다.

## 1. 결론부터

현재 방어 가능한 핵심 신규성은 **GNN, 디지털 트윈, counterfactual 또는 conformal prediction 각각**이 아니다. 다음 네 요소를 하나의 causal message-passing operator로 묶은 설계가 검증 대상이다.

1. fault label이나 `plant_mode`가 아니라 **관측 가능한 FDV feedback·CIP relay·유량·steam·전원·chemical 신호**로 material route를 여닫는다.
2. advective edge마다 관측 유량과 edge holdup을 이용해 `tau_e(t)=V_e/(Q(t)/3600)`를 계산하고, 과거 두 sample 사이를 **fractional interpolation**한다.
3. 압력·유량·명령·utility/control 관계에는 같은 수송지연을 강제로 적용하지 않고 **instantaneous hydraulic/control relation**으로 분리한다.
4. 현재값을 복사하지 않는 causal nominal-observer residual을 센서·observer 불확실성으로 정규화하고, mode-matched 정상 reference와의 representation constraint를 함께 사용한 뒤 위 graph에 전달한다.

따라서 가장 안전한 논문 기여 문장은 다음과 같다.

> FlowTwin-Guard는 합성 HTST 고장진단을 위해, 관측 가능한 운전신호로 경로를 조건화하고 edge별 물질 체류시간을 `V/Q` fractional delay로 표현하면서 비수송 관계를 즉시 전달하는 dual-relation graph operator를, mode-matched causal nominal-observer residual과 결합한다.

이것은 **검증할 가설**이지 현재 입증된 결론이 아니다. 특히 2026년 DSPR preprint가 physics-guided dynamic graph, residual stream, adaptive transport window를 이미 함께 다루므로 “최초의 동적 수송지연 graph”라는 넓은 주장은 방어할 수 없다. FlowTwin-Guard의 차별점은 learned channel window가 아니라 **명시적 edge `V/Q`**, **실제 관측 relay에 의한 route opening/closure**, **advective와 instantaneous relation의 분리**, **HTST 진단용 mode-matched nominal residual**의 좁은 결합이어야 한다.

## 2. 수학적으로 고정할 제안 연산자

Advective edge `e=(u,v)`의 message를 다음처럼 정의한다.

```text
tau_e(t) = V_e(profile) / (Q_obs(t) / 3600)
m_e^adv(t) = g_e(z_obs(t)) * W_e * Interp[h_u](t - tau_e(t))
```

- `g_e`는 FDV feedback, CIP relay, measured flow, steam, power, chemical concentration만 사용한다.
- 지연시각이 window 이전이면 첫 값을 복제하지 않고 `valid=false`, zero message로 처리한다.
- simulator-exact FIFO overlay와 P&ID nominal prior를 artifact에서 구분한다.

비수송 관계는 다음처럼 별도 처리한다.

```text
m_e^inst(t) = g_e(z_obs(t)) * W_e * h_u(t)
```

입력 표현은 다음 causal nominal residual이다.

```text
r_i(t) = (x_i(t) - xhat_i,N(t | x(<=t-1)))
         / sqrt(sigma_twin_i(t)^2 + sigma_sensor_i^2)
```

Observer는 `train_id`의 `N00`, `M01`, `M02`만으로 학습한 뒤 freeze한다. 일반 고장은 같은 counterfactual group의 `N00`, incomplete-cleaning `F12`는 mode-matched `M02`와 짝을 이룬다. 이 pairing metadata는 학습 목적과 평가 정렬에만 쓰며 inference input에는 들어가지 않는다.

이 네 식의 **결합 전체**가 기여 단위다. 각 식의 재료나 보조 loss 하나를 떼어 “새로운 방법”이라고 주장하지 않는다.

## 3. 1차 문헌 대비 신규성 경계

아래 링크는 제목·DOI·출판처를 출판사 또는 저자 원문에서 다시 확인한 항목만 사용했다.

| 선행연구 | 확인된 범위 | FlowTwin-Guard 주장에 주는 제약 |
|---|---|---|
| [DSPR: Dual-Stream Physics-Residual Networks for Trustworthy Industrial Time Series Forecasting, arXiv:2604.07393](https://arxiv.org/abs/2604.07393) | industrial forecasting에서 physics-residual stream, physics-guided dynamic graph, 학습된 channel별 adaptive temporal window를 결합한다. 저자 원문은 KDD 2026 accepted라고 표기한다. | “residual + dynamic graph + adaptive/flow-dependent delay”의 넓은 조합은 신규성이 아니다. DSPR은 가장 가까운 선행연구이며 반드시 본문 비교와 차이표에 들어가야 한다. |
| [Topology-Guided Graph Learning for Process Fault Diagnosis, IECR 2023, DOI 10.1021/acs.iecr.2c03628](https://pubs.acs.org/doi/10.1021/acs.iecr.2c03628) | process flowchart topology, attention, graph convolution과 graph explanation을 TEP 진단에 사용한다. | 정적 P&ID/flowchart-guided GNN 자체는 신규성이 아니다. |
| [Interaction-Aware Graph Neural Networks for Fault Diagnosis of Complex Industrial Processes, TNNLS 2023, DOI 10.1109/TNNLS.2021.3132376](https://ieeexplore.ieee.org/document/9655479/) | sensor interaction을 graph로 표현해 복합 산업공정 고장을 진단한다. | process fault diagnosis에 GNN을 쓰는 것 자체는 신규성이 아니다. |
| [Fusing logic rule-based hybrid variable graph neural network approaches…, ESWA 2024, DOI 10.1016/j.eswa.2023.121753](https://www.sciencedirect.com/science/article/pii/S0957417423022558) | switch·interlock 같은 discrete variable의 logic rule과 continuous variable을 graph diagnosis에 결합한다. | discrete relay/control 정보와 GNN의 결합 자체는 신규성이 아니다. 본 연구는 relay를 label proxy가 아니라 물리 route gate로 제한한다는 차이를 검증해야 한다. |
| [A propagation path-based interpretable neural network model…, Control Engineering Practice 2024, DOI 10.1016/j.conengprac.2024.105988](https://www.sciencedirect.com/science/article/pii/S0967066124001485) | fault propagation path를 GCN architecture에 넣어 chemical-process FDD와 해석을 수행한다. | graph propagation path와 해석가능성 자체는 신규성이 아니다. |
| [Integration of Digital Twin, Machine-Learning and Industry 4.0 Tools for Anomaly Detection: An Application to a Food Plant, Sensors 2022, DOI 10.3390/s22114143](https://www.mdpi.com/1424-8220/22/11/4143) | tube-in-tube 간접 식품 pasteurization plant에서 DT·IoT·ML anomaly detection을 결합한다. | 식품 pasteurizer, DT+ML, anomaly detection의 조합은 신규성이 아니다. |
| [HACCP with multivariate process monitoring and fault diagnosis techniques: application to a food pasteurization process, Food Control 2005, DOI 10.1016/j.foodcont.2004.04.008](https://www.sciencedirect.com/science/article/pii/S0956713504000921) | HTST pilot plant의 CCP에 multivariate process monitoring과 FDD를 적용한다. | HTST/HACCP 문맥의 FDD 자체는 신규성이 아니며, 본 ML을 HACCP 적합성 증명으로 표현할 수도 없다. |
| [A digital twin system for centrifugal pump fault diagnosis driven by transfer learning based on GCNs, Computers in Industry 2024, DOI 10.1016/j.compind.2024.104155](https://www.sciencedirect.com/science/article/pii/S0166361524000836) | pump digital twin의 합성자료, GCN, simulation-to-measurement transfer를 결합한다. | digital twin+GCN 및 합성자료 기반 graph diagnosis는 신규성이 아니다. FlowTwin-Guard에는 아직 sim-to-real transfer 증거도 없다. |
| [Uncertainty-aware deep learning for monitoring and fault diagnosis from synthetic data, RESS 2024, DOI 10.1016/j.ress.2024.110386](https://www.sciencedirect.com/science/article/pii/S0951832024004587) | physics model의 합성자료로 학습할 때 aleatoric·epistemic uncertainty를 진단모델에 반영한다. | 합성 FDD의 uncertainty-aware learning 자체는 신규성이 아니다. |
| [Counterfactual-augmented few-shot contrastive learning for machinery intelligent fault diagnosis, MSSP 2024, DOI 10.1016/j.ymssp.2024.111507](https://www.sciencedirect.com/science/article/pii/S0888327024004059) | 제한·불균형 표본의 machinery diagnosis에 counterfactual augmentation과 contrastive learning을 사용한다. | fault diagnosis에 counterfactual을 쓰는 것 자체는 신규성이 아니다. 현재 연구의 simulator-matched normal pair는 별도의 제한된 설계 선택이다. |
| [Causal Counterfactual Faithfulness Generation for Open-Set Fault Diagnosis…, TII 2025, DOI 10.1109/TII.2025.3598419](https://ieeexplore.ieee.org/document/11159595/) | graph representation과 causal generative counterfactual을 open-set industrial diagnosis에 사용한다. | counterfactual·causal이라는 단어만으로 신규성이나 인과성을 주장할 수 없다. 현재 D1은 open-set/OOD 평가가 아니다. |
| [Uncertainty-Aware Fault Diagnosis with Conformal Prediction, IFAC-PapersOnLine 2025, DOI 10.1016/j.ifacol.2025.09.092](https://www.sciencedirect.com/science/article/pii/S2405896325008535) | TEP와 CSTR fault classification에 conformal prediction set과 coverage를 적용한다. | fault diagnosis의 conformal prediction 자체는 신규성이 아니다. |
| [Reducing false alarms in fault detection: conformal prediction vs classical PCA/AE methods, Journal of Process Control 2025, DOI 10.1016/j.jprocont.2025.103495](https://www.sciencedirect.com/science/article/pii/S0959152425001234) | TEP에서 conformal thresholding과 고전 threshold를 false-alarm 관점에서 비교한다. | conformal threshold 또는 false-alarm control 자체는 신규성이 아니다. |

### 철회되어 근거에서 제외한 항목

`10.1016/j.psep.2026.109204`는 검색 캐시에 *Industrial fault diagnosis via topology inference and feature compensation*로 남아 있지만, 2026-07-27 확인한 [Crossref 등록 메타데이터](https://api.crossref.org/works/10.1016/j.psep.2026.109204)는 제목을 `WITHDRAWN:`으로 표시한다. 따라서 이 논문은 선행 성능, 방법 타당성 또는 신규성 판단의 근거에서 완전히 제외한다.

### 문헌검토가 허용하는 주장

- 허용: “검토한 선행연구와 비교할 때, FlowTwin-Guard가 검증하려는 차이는 explicit edge `V/Q` fractional transport, observable route closure/opening, instantaneous non-advective relation, mode-matched nominal residual의 결합이다.”
- 불허: “세계 최초”, “최초의 transport-delay GNN”, “최초의 food DT anomaly model”, “최초의 counterfactual/conformal FDD”.
- 제출 전 조건: 최소한 Scopus/Web of Science/IEEE Xplore/ACM DL/Crossref와 특허 검색을 날짜·검색식·제외사유와 함께 별도 systematic search appendix로 동결한다.

## 4. H1–H5 사전등록 가설

등록 지표의 약어는 다음과 같다.

- `M-F1`: `test_id` 20-class macro-F1, 높을수록 좋음
- `E-F1`: event F1, 높을수록 좋음
- `LAT`: detection-eligible event의 **mean horizon-penalized detection latency(s)**, 낮을수록 좋음. Miss에는 observable effect에서 event 종료까지 남은 horizon을 부여하고 `E-F1`·event recall·miss count와 항상 함께 보고
- `FA`: false-alarm **onsets** per negative hour, 낮을수록 좋음
- `UV`: 첫 post-effect alarm 전 누적 unsafe-forward L, 낮을수록 좋음
- `COV/SIZE/RC`: prediction-set marginal coverage, mean set size, selective risk–coverage curve
- `OOD`: energy AUROC, energy AUPR, energy FPR@95%TPR. 현재 ID-only D1에서는 세 항목 모두 `not_applicable`
- `EFF`: trainable parameters, training wall time, test rows/s

| 가설 | 사전 지정 비교 | 주 판단지표 | 반증 조건과 negative control |
|---|---|---|---|
| **H1 — 결합 연산자 유효성**: 전체 FlowTwin-Guard가 같은 입력·split·budget의 temporal/graph/twin baseline보다 ID 진단과 event trade-off를 개선한다. | `flowtwin_guard` 대 `tcn`, `causal_transformer`, `static_pid_gnn`, `dynamic_gnn`, `twin_residual_tcn`, `dspr_diagnostic_adaptation` | 공동 핵심 `M-F1`, `E-F1`; 안전 trade-off `FA`, `UV`; `EFF` 공개 | strongest baseline 대비 개선이 없거나, F1 향상이 더 큰 `FA`/`UV` 악화에 의존하면 기각. `no_status_relays`와 F13/F15 제외 결과에서 이점이 사라지면 일반 파형진단 기여로 주장하지 않음. |
| **H2 — 물질수송 시간**: edge별 flow/profile-dependent `V/Q` fractional delay가 zero 또는 fixed nominal delay보다 관측효과 정렬을 개선한다. | 전체 대 `no_delay`, `fixed_nominal_delay` | `LAT`, `UV`, `E-F1`, 보조 `M-F1` | near-nominal-flow strata에서는 차이가 작고 off-nominal strata에서 커져야 한다는 interaction을 사전 지정. 양 strata에서 무차별하거나 fixed가 우세하면 `V/Q` 기여를 기각. |
| **H3 — 관측 route gating**: FDV/CIP 상태에 따른 edge opening/closure가 static route보다 route-transition fault propagation을 잘 표현한다. | 전체 대 `static_route`; 보조로 `static_pid_gnn` | `E-F1`, `FA`, `UV`, route-transition strata의 `M-F1` | route가 안정된 negative-control 구간에서는 큰 이점이 없어야 하고, FDV/CIP transition에서 차이가 집중돼야 한다. 차이가 단순 relay-label shortcut으로 설명되면 기각. |
| **H4 — dual relation과 matched nominal residual**: advective delay와 instantaneous relation의 분리 및 causal nominal residual이 raw/static 표현보다 유효하다. | 전체 대 `single_relation`, `raw_only`, `no_uncertainty`, `no_counterfactual_loss`; baseline `twin_residual_tcn` | `M-F1`, `E-F1`, `LAT`; F12 mode-matched 결과는 별도 | `single_relation`과 차이가 없으면 relation 분리 기여를, `raw_only`/`twin_residual_tcn`과 차이가 없으면 graph-residual 결합 기여를 기각. `no_uncertainty`와 `no_counterfactual_loss`는 보조요소 검증이며 각각을 독립 신규성으로 승격하지 않음. |
| **H5 — 선택적 신뢰성**: validation-only conformal layer가 raw argmax보다 오류를 숨기지 않는 유용한 risk–coverage trade-off를 제공한다. | 전체 대 `no_conformal` | `COV`, `SIZE`, `RC`; `FA` 병기 | nominal 90% coverage만 보고 통과시키지 않는다. 목표 coverage를 달성하면서 set size와 risk–coverage가 유용해야 한다. 실패 시 conformal은 제거하거나 실패 결과로 보고하며, architecture 신규성에는 영향을 주지 않는다. |

H1은 논문의 주효과, H2–H4는 **왜** 효과가 발생하는지 검증하는 mechanistic 가설, H5는 배포 주장이 아닌 selective-output 평가다. H2 또는 H3가 반증되면 논문 제목과 초록에서 route-conditioned transport를 핵심 기여로 내세울 수 없다.

## 5. 정확한 비교모델과 9개 ablation

### 5.1 등록된 neural baselines

| 이름 | 격리하는 질문 |
|---|---|
| `tcn` | graph·twin 없이 causal temporal convolution만으로 충분한가? |
| `causal_transformer` | causal masked attention만으로 장기 의존성을 해결할 수 있는가? |
| `static_pid_gnn` | 고정 P&ID adjacency만으로 충분한가? |
| `dynamic_gnn` | 물리 route/delay 없이 sample-conditioned learned topology만으로 충분한가? |
| `twin_residual_tcn` | nominal-observer residual만으로 충분하고 graph operator는 불필요한가? |
| `dspr_diagnostic_adaptation` | closest prior의 dual trend/residual stream·physics-guided dynamic graph·adaptive temporal window로 충분한가? |

모든 neural baseline은 동일 29개 허용신호, causal window, profile split, class weights, heads, epoch budget과 validation-only calibration을 사용해야 한다. parameter count가 크게 다르면 matched-capacity sensitivity를 추가하되, 결과를 원래 등록 비교와 바꾸지 않는다.

Rule/EWMA/CUSUM은 별도 운영 기준선으로 유지한다. 이들은 ML을 이기기 위한 약한 모델이 아니라 기존 deterministic monitoring과의 trade-off를 보여주는 기준이다. `XGBoost`는 문서상 후보였지만 현재 `flowtwin_benchmark_contract.json`의 `registered_models`에는 없다. external-confirmatory 표에 넣으려면 **sealed test를 열기 전에** 계약 버전을 올려야 한다.

DSPR은 v0.2에서 이미 여섯 번째 baseline으로 등록됐다. 이 코드는 arXiv v3의 식·Algorithm 1에서 dual stream, prior-fused static/dynamic graph, channel-specific adaptive causal window, graph regularization을 추적했지만, 원 과제인 multi-horizon forecasting을 per-row six-output diagnosis로 바꾸고 TimeMixer를 causal rolling multi-scale encoder로 대체했다. 저자 구현이 공개되지 않은 상태에서 식을 독립 구현한 **closest-prior diagnostic adaptation**이며, author-code exact reproduction이나 원 논문의 forecasting 성능 재현이 아니다. 구체적 deviation은 각 checkpoint config의 executable metadata에 저장한다. DSPR은 FlowTwin의 route gate와 explicit edge `V/Q` delay를 사용하지 않는다.

그러므로 등록 행렬은 **FlowTwin-Guard 1 + neural baseline 6 + ablation 9 = 16 variants**다. Baseline 추가 또는 교체는 새 sealed external-confirmatory 실험의 test를 열기 전에만 contract version을 올려서 할 수 있다.

### 5.2 등록된 9개 ablation

| 정확한 이름 | 단 하나만 바꾸는 요소 | 연결 가설 | 해석 제한 |
|---|---|---|---|
| `no_delay` | 모든 advective delay를 0으로 설정 | H2 | temporal context 제거가 아니라 transport alignment만 제거해야 함 |
| `fixed_nominal_delay` | profile·관측유량 변화를 고정 nominal 값으로 대체 | H2 | off-nominal interaction이 핵심이며 평균점수만으로 결론내리지 않음 |
| `static_route` | observable FDV/CIP route gate를 고정 edge로 대체 | H3 | 모든 edge를 무차별 활성화하면 약한 straw ablation이 될 수 있으므로 정확한 edge 상태를 공개 |
| `single_relation` | hydraulic/control까지 advective relation으로 처리 | H4 | 의도적으로 잘못된 물리 가정에 대한 sanity check이지 경쟁 SOTA가 아님 |
| `raw_only` | nominal-observer residual을 제거하고 raw standardized signal 사용 | H4 | observer 학습도 함께 끄고 parameter/budget 차이를 보고 |
| `no_uncertainty` | sensor/twin denominator 없이 unnormalised residual 사용 | H4 | uncertainty 자체의 신규성 주장이 아니라 residual scaling 기여만 평가 |
| `no_counterfactual_loss` | counterfactual loss weight `0.10 → 0` | H4 | paired reference를 inference feature로 쓰지 않았는지 계속 감사 |
| `no_conformal` | prediction set·abstention 제거, validation anomaly threshold와 argmax만 사용 | H5 | raw accuracy 비교가 아니라 matched risk/coverage 비교 |
| `no_status_relays` | `power_good_signal`, `temperature_sensor_quality_ok` 제외 | H1 음성대조 | F13/F15 direct-status 이점을 전체 모델 성능으로 포장하지 않음 |

아홉 개 모두 같은 split과 `20260727`, `20260728`, `20260729` seed에서 실행하고 실패 seed도 보존한다. 한 번에 두 축을 바꾼 결과는 등록 ablation으로 세지 않는다.

## 6. 통계 단위와 분석 계약

### 6.1 독립 단위

**통계적 독립 단위는 `plant_profile_id`다.** row, overlapping window, event, episode, counterfactual pair 및 training seed는 독립 표본 수가 아니다.

- profile 안의 20 class × replicate × time row는 반복측정이다.
- training seed 3개는 optimizer variability를 보여주는 technical repeat이지 `n=3`의 공정 표본이 아니다.
- model 차이는 같은 profile·split·seed에서 paired difference로 계산한다.
- 95% interval은 profile을 outer cluster로 resample하는 paired hierarchical bootstrap으로 계산한다. seed는 같은 resample 안에서 평균하거나 inner technical-repeat 층으로 처리한다.
- row bootstrap, window-level t-test, 수백만 row를 표본수로 쓴 p-value는 금지한다.

Benchmark contract v0.2의 `aggregation.inferential_unit`은 이제 `plant_profile_id`로 고정돼 있다. Seed는 `technical_repeat`로만 요약하고, runner는 공통 profile에서 모델 차이를 짝지은 뒤 profile을 outer cluster, seed를 inner technical-repeat로 다루는 결정적 hierarchical bootstrap을 발행한다. D1-pilot의 test profile 3개 한계는 이 교정으로 사라지지 않으므로 D1은 pilot이지 external-confirmatory evidence가 아니다.

### 6.2 현재 D1·D2의 독립 표본 한계

현재 `D1-pilot`은 4,320,000 rows, 1,200 episodes, 12 profiles이며 split은 train 7 / validation 2 / test 3 profiles다. 300개 test episode가 있어도 독립 test unit은 **3 profiles**뿐이다. 이 수로는 수백만 row 기반의 좁은 신뢰구간이나 강한 일반화 주장을 만들 수 없다.

현재 D1에는 `test_ood_profile`이 없으므로 contract의 `combined-test` event 지표도 사실상 `test_id`만 집계한다. OOD 세 지표에는 0이나 ID 대체값을 넣지 않고 `not_applicable`로 기록한다.

`D2-ood-dev`는 별도로 생성·분할·감사가 끝난 합성 support-shift 개발 set이다. ID 12 + OOD 4 = 16 profiles, 48 counterfactual groups, 960 episodes, 1,728,000 rows이고 split은 train 7/420, validation 2/120, test-ID 3/180, test-OOD 4/240 profiles/episodes이다. `OOD_LOW_FLOW`과 `OOD_HIGH_FLOW_WARM_FEED`가 각 2 profiles이며 감사 30/30을 통과했다. Domain verifier는 profile config hash를 실제 33개 physical parameter에서 재구성하고, profile 값과 계약 range, domain 사이 최소 한 파라미터의 strict closed-interval support gap을 확인한다. 이는 synthetic OOD 계약 무결성이지 현장 OOD 타당성이 아니다. 전체 16-variant×3-seed matrix는 아직 미실행이므로 OOD endpoint는 현재 성능 결론이 아니다.

제출용 external-confirmatory dataset의 profile 수는 D1 test effect를 보지 않고, 최소 관심효과와 원하는 interval 폭을 이용한 prospective simulation/power analysis로 정한다. 새 profile family와 noise/counterfactual seed family는 학습자료와 분리하고 test 종료까지 봉인한다.

### 6.3 지표 계산 원칙

- 중첩 window의 모든 supervised/transport/counterfactual training loss와 exhaustive evaluation은 `valid_mask & eval_mask`를 사용한다. `valid_mask`는 causal context/padding 유효성, `eval_mask`는 row의 비중복 owning loss region을 뜻한다.
- Event-aware sampler의 onset/middle/end landmark는 window안에 존재하는 것만으로 충분하지 않고 반드시 그 window의 owning `eval_mask` region에 속해야 한다. Target-bearing train episode 중 한 개라도 target row가 loss region에서 누락되면 run을 중단한다.
- `M-F1`은 `test_id`에서만 고정 20-class vocabulary로 profile별 계산하고 absent-class 처리 규칙을 사전 고정한다.
- `LAT`는 성공 event 지연의 평균이 아니라 miss에 남은 observable-event horizon을 부여한 mean horizon-penalized endpoint다. 그래도 survivorship/trade-off를 숨기지 않도록 event recall, `E-F1`, miss count, `UV`와 분리하지 않는다.
- `FA`는 alarm step 수가 아니라 **새 alarm onset 수/negative hour**다. 기존 smoke의 `false-alarm steps/hour`와 직접 비교하지 않는다.
- `UV`는 volume-weighted safety proxy이며 제품안전 판정이 아니다.
- conformal은 validation-ID의 counterfactual-group/observable-mode block별 true-class nonconformity와 ID energy 최대값으로 적합한다. Coverage만 높이려고 모든 class를 담는 해를 막기 위해 `COV`, `SIZE`, `RC`를 함께 본다.
- D1 validation은 profile 2개이므로 `COV`는 **empirical synthetic row coverage**로만 보고하며, held-out profile shift에 대한 finite-sample coverage guarantee는 하지 않는다.
- 효율 지표와 센서 의존성은 성능표와 같은 위치에 공개한다.

## 7. Negative controls와 falsification tests

등록 아홉 ablation 외에 다음 분석을 **결과를 보기 전에** protocol에 고정한다.

| 대조 | 기대 패턴 | 반대 결과의 의미 |
|---|---|---|
| pre-effect 및 정상 `N00/M01/M02` 구간 | anomaly score와 alarm onset이 낮아야 함 | fault schedule·episode time·reference metadata 누수 가능성 |
| near-nominal flow 대 off-nominal flow | `fixed_nominal_delay`와의 차이는 off-nominal에서 더 커야 함 | `V/Q`가 아니라 추가 parameter/capacity 효과일 가능성 |
| stable route 대 FDV/CIP transition | `static_route`와의 차이는 transition에서 더 커야 함 | relay shortcut 또는 일반 regularization 효과일 가능성 |
| F13/F15 포함 대 제외 | status-aware 성능과 sensor-only 성능을 분리 | deterministic status mirror로 만든 쉬운 탐지를 전체 성능으로 과대평가 |
| same profile의 matched counterfactual | effect 이전 representation은 가깝고 effect 이후만 분리 | onset/label을 외우거나 pair alignment가 잘못됐을 가능성 |
| future-input perturbation | 과거 logits 불변 | causal inference 위반 |
| forbidden/oracle feature audit | 0개 | 실험 전체 무효 |

Label permutation, route-signal permutation, edge-volume misspecification stress는 강한 sanity check가 될 수 있지만 현재 아홉 ablation에는 포함되지 않는다. 수행하려면 external-confirmatory 실험 전에 별도 `falsification_tests`로 등록하고, 본 ablation과 섞지 않는다.

## 8. 출판 게이트

Runner tier는 `development` → `pilot` → `protocol_complete_synthetic`이다. 최상위 `protocol_complete_synthetic`은 전 16개 variant×3 seeds, exact frozen CPU budget, 검증된 `test_id`/`test_ood_profile` domain을 모두 갖춘 **합성 프로토콜 완료**만 뜻한다. 독립 외부 데이터, prospective preregistration, profile-level power 근거와 현장 검증을 자동으로 만들지 않으며, runner는 절대 `external_confirmatory`라는 판정을 부여하지 않는다.

| Gate | 통과 조건 | 2026-07-27 상태 |
|---|---|---|
| **G0 문헌·claim gate** | DSPR을 포함한 systematic database/patent search, 철회논문 제외, 검색식·일자 공개 | **미통과** — 1차 targeted matrix는 검산했지만 systematic/patent search는 아직 없음 |
| **G1 데이터 무결성** | checksum, schema, forbidden-field, profile/seed/counterfactual isolation, row alignment 감사 0 failure | **통과(개발 data)** — D1-pilot·D2-ood-dev 각 30/30 audit pass |
| **G2 사전등록 완전성** | exact contract, FlowTwin 1 + neural baseline 6 + ablation 9 = 16 variants, three seeds, fixed budget; test label로 선택하지 않음 | **미통과** — 전체 benchmark 미실행; D1은 ID-only pilot |
| **G3 독립 표본 적정성** | profile-level power/precision 설계와 sealed test profiles 확보 | **미통과** — D2도 test-ID 3, test-OOD 4 profiles의 development set이며 prospective power 근거가 없음 |
| **G4 주효과 H1** | strongest registered baseline 대비 profile-paired `M-F1`·`E-F1` 개선을 보이고 `FA`·`UV` 악화로 설명되지 않음 | **v0.3 개발 결과에서 미충족** — Hybrid ID macro-F1 `0.68129`는 1위지만 전체/OOD macro-F1은 DSPR보다 낮고, operational 전체 event-F1은 FlowTwin보다 낮음; opened-D2·1 seed post-hoc이므로 최종 판정 아님 |
| **G5 mechanism H2–H4** | delay, route, dual relation, residual의 사전 예상 effect direction이 profile strata와 paired interval에서 재현 | **미평가** |
| **G6 selective H5** | block-max validation calibration 후 사전 고정한 empirical synthetic row-coverage tolerance를 만족하고 유용한 set size/risk–coverage를 보임 | **v0.3 개발 결과에서 미충족** — Hybrid/FlowTwin/TCN/DSPR 평균 set `12.60/15.48/13.22/15.06`, 네 모델 모두 `DIAGNOSE=0`; F05 recall도 모두 0 |
| **G7 OOD/open-set** | 학습 전에 정의·봉인된 OOD profile set에서 energy AUROC/AUPR/FPR@95%TPR 평가 | **v0.3 개발 평가·게이트 미충족** — Hybrid physical score AUROC/FPR95 `0.55268/0.91749`, FlowTwin `0.76773/0.93618`; opened-D2·1 seed·부분집합이며 실용적 FPR95가 아님 |
| **G8 현장·안전** | 독립 실제 설비 자료, sensor calibration, as-built P&ID/PLC, 제품안전/HACCP 검증 | **미통과/현재 범위 밖** |

H1의 exact superiority/non-inferiority margin과 H5 coverage tolerance는 현재 contract에 수치로 고정돼 있지 않다. D1 결과를 본 뒤 margin을 정하면 안 된다. 새 sealed dataset 전에 공정적으로 의미 있는 최소효과와 허용 `FA`/`UV` 악화 한계를 문헌·운영근거로 등록한다.

## 9. 현재 증거 상태

### 확인된 것

- 22-node/23-edge graph, observable route gate, profile edge volume, fractional delay, causal nominal observer, hierarchy heads와 validation-only conformal 실행경로가 구현돼 있다.
- FlowTwin-Guard 1개, baseline 6개(가장 가까운 선행연구 DSPR diagnostic adaptation 포함), ablation 9개의 16-variant registry와 실행 runner가 구현돼 있다. DSPR은 exact author-code reproduction이 아니다.
- fractional interpolation과 gradient, route gate, future invariance, forbidden-feature 차단, `valid_mask & eval_mask` loss ownership/event-landmark region, block-max validation calibration에 대한 회귀시험이 있다.
- D1-pilot은 12 ID profiles, 1,200 episodes, 4,320,000 rows로 생성됐고 train/validation/test profile isolation을 포함한 감사 30/30을 통과했다.
- `D1-pilot-cache-v0.3`은 dataset/split/checksum, episode materialization, train-only scaler와 `ml_pipeline_common.py`를 포함한 materialization source SHA-256을 묶고 불일치 cache 로드를 거부한다.
- D2-ood-dev는 ID/OOD 16 profiles·48 groups·960 episodes·1,728,000 rows와 `OOD_LOW_FLOW`/`OOD_HIGH_FLOW_WARM_FEED` domain을 생성했고, profile hash·33-parameter range·strict support-gap 검증을 포함한 감사 30/30을 통과했다.
- `D2-ood-dev-cache-v0.3`는 960 episodes/1,728,000 rows, train-only 756,000 scaling rows와 `ml_pipeline_common.py`를 포함한 materialization source SHA-256을 묶었다.
- D2 frozen-budget 3-model×1-seed development run에서 FlowTwin/TCN/DSPR의 macro-F1은 0.445/0.563/0.586, event-F1은 0.019/0.036/0.257이었다. FlowTwin OOD AUROC 0.811은 가장 높았지만 FPR95 0.951과 347.85 false-alarm onsets/h를 동반했다.

### v0.3 W96·1-seed opened-D2 post-hoc 결과

네 모델 diagnostic은 `FlowTwin-v03-D2-candidate-matched-dev`, `FlowTwin-v03-D2-flowtwin-operational-dev`, `FlowTwin-v03-D2-tcn-raw-diagnostic-dev`, `FlowTwin-v03-D2-dspr-operational-dev`에 각각 checksum으로 고정했다. Row threshold는 validation에서 적합했으며, 아래 event/FA/unsafe는 state-machine 전 combined-test 지표다. `전체` macro-F1은 ID/OOD row를 합친 descriptive pooled 값이지 profile-level 추론 결과가 아니다.

| 모델 | parameter | 전체 / ID / OOD macro-F1 | validation-fitted row event-F1 | row-threshold FA h⁻¹ | row-threshold unsafe L | OOD AUROC / FPR95 |
|---|---:|---:|---:|---:|---:|---:|
| FlowTwin-Hybrid v0.3 | 58,315 | 0.56218 / **0.68129** / 0.53188 | 0.03452 | 186.51 | 414.37 | 0.55268 / 0.91749 |
| FlowTwin-Guard | 42,508 | 0.42491 / 0.55566 / 0.37698 | 0.02686 | 245.83 | 0.00 | **0.76773** / 0.93618 |
| TCN | 15,379 | 0.58313 / 0.64073 / 0.54890 | 0.02222 | 285.31 | 56.31 | 0.45451 / 0.98143 |
| DSPR adaptation | 54,166 | **0.59399** / 0.61125 / **0.58642** | **0.31215** | **5.24** | 4,081.49 | 0.57475 / 0.93870 |

Operational gate의 최대 profile FA `≤5 h⁻¹`, 전체 recall `≥0.8`, 각 eligible-profile recall `≥0.5`를 동시 적용한 결과는 다음과 같다.

| 모델 | validation 유효/전체 | validation event-F1 / recall | 최대 profile FA h⁻¹ | test operational event-F1 | test FA h⁻¹ | test unsafe L |
|---|---:|---:|---:|---:|---:|---:|
| FlowTwin-Hybrid v0.3 | 4/75 | 0.88889/0.80808 | 3.40089 | 0.87690 | 5.744 | 436.18 |
| FlowTwin-Guard | 4/75 | 0.89583/0.86869 | 4.29159 | **0.90040** | 7.770 | **10.51** |
| TCN | **0/75** | — | — | **미평가** | — | — |
| DSPR adaptation | 8/75 | 0.89362/0.84848 | 3.88673 | 0.88438 | **3.276** | 3,674.27 |

표준 보고서의 candidate-minus-comparator profile-paired bootstrap 기술값은 Hybrid의 ID macro-F1 차이를 DSPR 대비 `+0.07521` (95% interval `[0.06533, 0.09403]`), FlowTwin 대비 `+0.10170` (`[0.05081, 0.13233]`), TCN 대비 `+0.02470` (`[-0.04666, 0.08188]`)로 요약한다. 반면 operational event-F1 차이는 DSPR 대비 `-0.00244` (`[-0.04738, 0.03362]`), FlowTwin 대비 `-0.01786` (`[-0.05789, 0.01599]`)로 모두 0을 포함한다. 이 구간은 ID 3 profiles, operational 7 profiles, 기술 반복 1 seed, 이미 열린 D2의 10,000회 profile-cluster bootstrap 기술일 뿐이다. ID 부분 이득이 operational event-F1·FA·unsafe trade-off를 함께 앞서지 못하므로 우월성 근거로 쓰지 않는다.

TCN은 `ml_results/FlowTwin-v03-D2-tcn-alarm-feasibility-dev`에 `test_iterator_constructed=false`, `test_rows_seen=0`로 고정됐다. 이는 성능표에서 실패 variant를 누락한 것이 아니라 사전 제약을 유지한 fail-closed 결과다. Operational test 표의 세 모델 비교는 gate 통과에 조건부이며 TCN을 넘어선 3-model 순위로 해석해서는 안 된다.

Hybrid/FlowTwin의 validation 최대 profile FA는 test profile에서 `9.42/9.09 h⁻¹`로 상승해 제약 일반화에 실패했다. DSPR test-profile 최댓값은 `4.18 h⁻¹`였지만 operational recall `0.81322`와 unsafe `3,674.27 L`의 trade-off가 크다. Hybrid은 ID macro-F1 1위일 뿐 전반적 우월성을 보이지 않았고, `F05` recall은 네 모델 모두 `0`, Hybrid `F09` recall도 `0`, conformal `DIAGNOSE`도 네 모델 모두 `0`이다. 병렬 CPU 경합 때문에 timing은 비교하지 않고 parameter count만 효율 정보로 사용한다.

### 아직 확인되지 않은 것

- 전체 등록 baseline 대비 우월성
- 아홉 ablation의 예상 effect direction
- profile-level uncertainty와 재현성
- 유용한 conformal set size·singleton diagnosis·risk–coverage 달성
- OOD/open-set full-matrix·다중-seed 재현성과 실용적 FPR95
- simulator-to-real transfer
- 현장 진단 정확도, 살균 유효성, 제품 안전 또는 HACCP 적합성

기존 4-profile smoke의 row macro-F1 `0.353002`, nominal-90% coverage `0.850208`, singleton `DIAGNOSE=0`은 성능 성공이 아니다. event recall 17/17도 `3430.697674 false-alarm steps/negative-hour`와 함께 발생했으므로 안전성 증거가 아니다. smoke는 구현·artifact 경로의 실패를 포함한 점검 기록으로만 인용한다.

또한 `ml_results/FlowTwin-Guard-D1-seed20260727/`의 test-ID row accuracy `0.26423`과 nominal-90% empirical row coverage `0.64940`은 v0.2 전 **pre-audit direct run**이다. 이후 loss mask ownership, event landmark owning region, block-max conformal, cache provenance, DSPR 비교, profile aggregation과 tier 규칙이 교정됐으므로 pipeline-only 기록으로만 보존하고 모든 v0.2 가설 검정·baseline/ablation 비교·출판 성능표에서 제외한다. `ml_results/FlowTwin-Benchmark-D1-core-dev-v0.2/`도 구 cache v0.2를 사용한 3-variant×1-seed reduced **development** run이며, cache source-provenance 누락을 교정한 v0.3 계약의 성능 근거로 승격하지 않는다.

## 10. 예상 reviewer 위협과 선제 대응

| 위협 | 왜 치명적인가 | 논문 전에 필요한 대응 |
|---|---|---|
| DSPR과의 중복 | adaptive delay, dynamic graph, residual이라는 상위 개념이 겹침 | explicit edge `V/Q`, route closure, dual relation, diagnosis/matched residual의 차이를 식·표·직접비교로 격리 |
| 자기 simulator에서 자기 model 평가 | generator inductive bias를 model에 그대로 넣으면 당연히 유리할 수 있음 | edge-volume/flow misspecification, alternative simulator 또는 독립 generator, 최종적으로 실제 data 외부검증 |
| simulator-exact FIFO overlay | baseline에 없는 privileged structural prior일 수 있음 | exact/prior 구분 공개, `fixed_nominal_delay`, volume perturbation, 모든 비교모델 입력계약 공개 |
| route relay shortcut | FDV/CIP/control relay가 class를 직접 드러낼 수 있음 | stable/transition strata, relay permutation sanity test, `no_status_relays`, class별 error analysis |
| direct status mirror | F13/F15가 power/sensor truth mirror로 너무 쉬움 | status-aware와 sensor-only 표를 분리하고 F13/F15 제외 macro-F1 공개 |
| counterfactual oracle | 현장에는 완벽히 matched 정상 trajectory가 없음 | training-only synthetic auxiliary임을 명시하고 `no_counterfactual_loss`, unmatched/noisy-pair sensitivity 수행 |
| pseudoreplication | 4.32M rows가 독립 표본처럼 보임 | profile cluster만 `n`으로 사용; test 3 profiles인 D1은 pilot로 제한 |
| ablation straw man | `single_relation`이나 all-edge `static_route`가 의도적으로 나쁠 수 있음 | 경쟁 baseline과 mechanistic sanity ablation을 구분하고 static-route 정의·parameter 수 공개 |
| conformal guarantee 과장 | profile shift는 calibration exchangeability를 깨뜨릴 수 있음 | marginal/stratified coverage와 set size 공개; 실패를 숨기지 않고 OOD guarantee 금지 |
| validation 운전점 과적합 | Hybrid/FlowTwin의 `≤5 h⁻¹` validation FA 제약이 test profile에서 깨짐 | 새 profile의 nested calibration/test, 운전점 봉인, profile-level FA 구간과 gate-failure 전수 공개 |
| 실패 variant 생존편향 | TCN `0/75`를 빼고 operational 3-model만 순위화하면 제약 성공률을 과대평가 | diagnostic 4-model, validation feasibility 4-model, conditional operational 3-model 표를 분리 |
| latency survivorship | miss가 많은 모델이 빠르게 보일 수 있음 | E-F1, miss, LAT, UV를 공동 보고 |
| test-set 적응 | baseline/ablation을 D1 test 결과 뒤 수정하면 external-confirmatory가 아님 | D1을 pilot로 선언하고 코드·contract·seed를 새 sealed dataset 전에 tag |
| multiplicity | 6 baselines×9 ablations×다수지표에서 유리한 결과 선택 가능 | H1–H5와 metric hierarchy 고정, 전 16 variants/seeds/failures 공개, profile-paired interval 사용 |
| DT·causal 용어 과장 | 현장동기화/식별이나 causal intervention 증거가 없음 | `engineering surrogate`, `causal computation`으로 한정; “validated digital twin”, “causal discovery” 금지 |

## 11. 금지 주장

다음 문구는 현 증거로 쓰지 않는다.

- “세계 최초”, “완벽한 모델”, “SOTA”
- “최초의 graph-based/transport-delay/digital-twin/counterfactual/conformal fault diagnosis”
- “high-fidelity 또는 현장 검증된 digital twin”
- “fault의 원인을 인과적으로 발견했다” 또는 “causal graph를 학습했다”
- “OOD/open-set 일반화가 확인됐다”
- “90% conformal이 profile shift에서도 90% 안전을 보증한다”
- “100% event recall이 안전을 증명한다”
- “살균 유효성, 미생물 안전, 제품 release, SAFE 판정 또는 HACCP 적합성을 보장한다”
- “실제 공장 정확도” 또는 “sim-to-real 성능”
- “CIP edge의 지연도 exact physical transport다”
- row/window를 독립 표본으로 계산한 유의성 주장
- F13/F15 direct-status 결과를 일반 sensor-only 진단성능으로 합산한 주장
- 현재 ID-only D1에서 산출한 OOD AUROC/AUPR/FPR95

허용되는 결과 문장은 다음 형식으로 제한한다.

> Profile-disjoint synthetic HTST benchmark에서, 사전등록된 입력·학습 budget 아래 FlowTwin-Guard와 기준선의 profile-paired 차이를 평가했다. 결과는 simulator 내부 타당성에 한정되며 현장 정확도, 제품안전 또는 HACCP 적합성을 의미하지 않는다.

## 12. 논문 구성 권고

권장 제목은 과장된 `digital twin`이나 `safety guarantee`보다 실제 기여를 앞에 둔다.

> **FlowTwin-Guard: Observable Route-Gated V/Q Transport Graphs with Nominal-Observer Residuals for Synthetic HTST Fault Diagnosis**

본문 순서는 다음이 가장 방어적이다.

1. 문제: route switching과 flow-dependent residence time이 있는 HTST 진단
2. 선행연구: HTST FDD, food DT, process GNN, adaptive-delay graph, counterfactual, uncertainty/conformal을 분리
3. 방법: observable gate, exact/prior volume provenance, `V/Q` interpolation, dual relation, nominal residual
4. protocol: profile split, oracle ban, baseline fairness, H1–H5, nine ablations
5. 결과: profile-level paired effect, event/safety-proxy/selective/efficiency를 함께 공개
6. falsification: flow/route strata, relay removal, misspecification, failed classes/seeds
7. 한계: D1 ID-only·D2 synthetic support-shift와 작은 profile 수, no external-OOD/field/HACCP claim

최종적으로 논문 가치가 생기는 조건은 모델이 복잡하다는 사실이 아니라, **H2와 H3가 사전 예측한 구간에서만 선택적으로 나타나고, H1의 이점이 강한 temporal·graph·twin baseline 및 direct-status 제거 뒤에도 유지되며, 그 대가가 false alarm과 unsafe-volume 악화가 아님을 profile 단위로 보이는 것**이다. 현재는 이 검증을 실행할 코드·데이터 계약까지 준비된 단계이며, 신규성·우월성·현장성은 아직 미입증이다.
