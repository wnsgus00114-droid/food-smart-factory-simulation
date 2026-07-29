# HTST 시뮬레이터 기반 ML 실험 설계

> 문서 상태: D1-pilot·D2-ood-dev 생성/분할/감사/cache 완료, FlowTwin benchmark protocol `0.2.0`·16-variant registry 구현, 별도 v0.3 W96·1-seed opened-D2 post-hoc 개발 평가 완료; 확정 결과는 우월성·confirmatory 근거에서 제외
> 적용 대상: HTST 공정 코어 `2.2.0` + 비바이오 통합 실행기 `3.0.0`
> D1 ML 계약 버전: `2.2.0` / FlowTwin protocol: `0.2.0` / runner: `0.3.0` / cache materialization: `0.3.0` / D3 계약 버전: `1.1.0`
> 최근 갱신: 2026-07-27 (KST)
> 범위: 합성 시뮬레이션 데이터의 연구 설계와 stdlib 기반 D1/D2/D3 생성·분할·누수감사·기준선. 바이오 target과 실제 공장·규제 적합성 검증은 포함하지 않는다.

## 1. 목적과 판정 경계

이 문서의 목적은 HTST 시뮬레이터를 단순 시나리오 재생기가 아니라, 다음 ML 문제를 반복 실험할 수 있는 벤치마크 생성기로 사용하는 것이다.

1. 공정 이상 탐지와 고장 분류
2. CCP 이탈·회송·냉각 이상의 조기경보
3. 계측하기 어렵거나 고장이 잦은 상태의 soft sensor 추정
4. 열교환기 오염의 건강도 추정, 예지보전, RUL 예측
5. 학습 범위 밖 운전조건과 미지 고장에 대한 OOD 탐지
6. 안전 인터록을 침범하지 않는 에너지·회송량·제어응답 최적화

모든 합격 기준은 **시뮬레이터 내부 연구 게이트**다. 이를 통과해도 실제 제품 안전, HACCP 적합성, 설비 고장 예측 성능 또는 공장 배포 가능성이 입증되는 것은 아니다. 바이오 모델과 바이오 target은 사용자 결정에 따라 제외한다. ML은 FDV, 압력차, 최소 체류시간 같은 결정론적 안전 인터록을 대체하지 않는다.

## 2. 현재 자료의 사용 가능 범위

`results/*.csv` 같은 단일 설정 시나리오 출력은 시뮬레이터 회귀·보고용 자료다. 다음 이유로 그대로 학습 데이터로 사용하지 않는다.

- 시나리오당 에피소드가 1개라 모델이 동역학보다 고정 파형을 외울 수 있다.
- 고장 시작시각과 크기가 고정돼 있어 절대시간만으로도 라벨을 맞힐 수 있다.
- `scenario`, `actual_safe`, `control_sensor_bias_c`, `safety_sensor_bias_c`, `heater_capacity_factor`, `regenerator_factor`는 정답 또는 정답 생성 인자다.
- 인접한 0.5초 행을 무작위 분할하면 사실상 같은 시계열 조각이 train/test에 동시에 들어간다.
- v2는 제어용·안전용 온도센서와 연속 FDV 위치를 분리했지만, 단일 설정·단일 seed 출력만으로는 편향·drift·dropout·밸브 고장의 다양한 크기와 형태를 학습하기에 부족하다.

따라서 단일 시나리오 결과는 `EXP-HTST-ML-000`의 스키마·재현성 점검에만 사용한다. 실제 ML 학습은 별도 생성기가 고장 시작시각, 지속시간, 심각도, 설비 파라미터와 난수 시드를 에피소드별로 무작위화한 `D1` 이상에서 시작한다.

## 3. v2.2 신호 계약

### 3.1 입력에 사용할 수 있는 관측 신호

실제 PLC/히스토리언에서 관측 가능하다는 가정 아래 다음을 후보 입력으로 둔다. 각 실험은 아래 신호군 중 하나를 명시해야 한다.

| 신호군 | 입력 후보 | 용도 |
|---|---|---|
| `S0-core` | `control_temp_sensor_c`, `safety_temp_sensor_c`, `measured_flow_l_h`, `steam_valve`, FDV command/position feedback/error, `product_temp_sensor_c` | 최소 센서·연속 밸브 피드백 기준선 |
| `S1-pressure` | `S0-core` + `raw_pressure_sensor_bar`, `pasteurized_pressure_sensor_bar`, `measured_differential_pressure_bar`, `booster_pump_speed_fraction`, `leak_detector_signal_fraction` | 부스터펌프·압력역전·누설 탐지 |
| `S2-derived` | `S1-pressure` + `preheat_temp_sensor_c`, `sensor_disagreement_c`, 평균·fastest-path 추정 체류시간, `maximum_safe_flow_l_h`, `cip_chemical_concentration_pct`, `cip_cycle_active`, CIP conductivity/pH | 측정값에서 계산하거나 PLC·CIP 계측이 제공한다고 가정한 신호 |
| `S3-context` | `S2-derived` + `cip_release_permissive`, `post_cip_conductivity_proxy_ms_cm`, `post_cip_ph_proxy`, `restart_product_interface_signal_fraction`, `power_good_signal`, `temperature_sensor_quality_ok` | 재기동 interface와 전원·센서 상태 relay까지 포함한 현재 최대 허용 신호군 |

`plant_mode`와 시간·episode 식별자는 조인과 mode별 평가를 위한 context로 저장하지만 현재 기본 모델 입력 allowlist에는 포함하지 않는다. mode-aware 실험에서 `plant_mode`를 쓰려면 계약 버전을 올리고 해당 ablation을 명시한다. `heat_kw`처럼 참값 물성·참유량으로 계산된 신호는 `results/schema.json`의 `oracle` 역할이며 입력에 넣지 않는다. 단일 진실원은 `ml_contract.json`이다.

관측 상태 신호의 의미는 다음과 같이 고정한다.

| 신호 | 계약상 의미 | 금지되는 해석 |
|---|---|---|
| `cip_cycle_active` | CIP controller/recipe가 내보내는 활성 relay. 기본 baseline의 생산/CIP gating에 사용 | `plant_mode` 진실값을 입력했다고 간주 |
| `cip_conductivity_proxy_ms_cm`, `cip_ph_proxy` | 설치 가능한 conductivity/pH 계측을 표현한 합성 관측 proxy | 숨은 soil·잔류 chemical 참값이나 실제 센서 교정 결과로 해석 |
| `post_cip_conductivity_proxy_ms_cm`, `post_cip_ph_proxy`, `restart_product_interface_signal_fraction` | 재기동 interface의 conductivity/pH/product-fraction 계측을 표현한 합성 proxy | 제품 조성·잔류 chemical·surface hygiene truth로 해석 |
| `cip_release_permissive` | 위 interface proxy에서 만든 PLC형 재기동 release relay | 독립 세정검증이나 제품출하 판정으로 해석 |
| `power_good_signal` | 배포 시 관측 가능한 PLC power-status interface를 표현. 현재 합성값은 `power_available`을 결정적으로 그대로 mirror | 독립 전원 계측이나 sensor-only 근거로 해석 |
| `temperature_sensor_quality_ok` | 배포 시 관측 가능한 device/channel-status interface를 표현. 현재 합성값은 `sensor_available`을 결정적으로 그대로 mirror | 독립 센서 교정·정확도 증명이나 sensor-only 근거로 해석 |

두 status relay는 F13 power failure와 F15 sensor dropout을 직접 드러내 탐지를 쉽게 만든다. 따라서 이를 쓴 결과는 `direct-status`/status-aware baseline으로 별도 표기하고, 온도·유량·압력 파형만으로 탐지하는 sensor-only ablation에서는 두 필드를 제외한다.

### 3.2 입력으로 금지할 privileged/oracle 신호

다음 값은 라벨 생성, 시뮬레이터 진단, 오차 분석에만 쓴다.

- `scenario`, `fault_type`, 고장 시작·종료시각, 고장 심각도
- `control_sensor_bias_c`, `safety_sensor_bias_c`와 개별 센서의 실제 주입 편향
- `heater_capacity_factor`, `regenerator_factor`, `cooler_factor`, `cleaning_effectiveness_factor`
- 실제 `sensor_available`, `valve_travel_time_factor`, `valve_leakage_fraction`, 내부 FDV 실제 위치와 그 truth에서 직접 만든 `fdv_actual_forward`
- balance tank의 실제 volume/temperature/pass/risk/chemical/product moment, CIP protein/mineral soil·제거량·잔류 chemical·surface hygiene·완료판정
- `campaign_id`, phase context, transition drain/addition, 실제 routed/forward product·chemical·hygiene fraction과 누적 부적합량
- shadow regenerator/heater/cooler의 내부 유체·wall 온도, 유효 UA와 에너지 보존오차
- 실제 유량, 실제 온도, 실제 체류시간 등 센서 오차가 적용되기 전의 참값
- 실제 `differential_pressure_bar`. 압력차 입력은 두 센서에서 얻은 `measured_differential_pressure_bar`만 허용
- `fouling_index`, `leak_fraction`, `contamination_risk`, `actual_safe`, `unsafe_forward_l`
- `alarm_unsafe_forward`, precomputed `alarm_count`/`observable_alarm_count`. 개별 진단 알람도 현재 기본 allowlist에서는 제외
- `time_s`, `time_start_s`, `episode_id`, `plant_profile_id`, `counterfactual_group_id` 같은 조인·분할용 context
- 미래의 FDV 상태, 미래 알람, 미래 정비시각
- 라벨을 직접 계산한 값 또는 그 단순 단조변환

예외는 해당 값을 **예측 목표**로 사용하는 경우뿐이다. 예를 들어 `fouling_index`를 건강도 라벨로 사용할 때 입력에서는 제거한다.

### 3.3 권장 데이터 구조

CSV 한 파일에 관측값과 정답을 섞지 않고 다음 4개 논리 테이블로 분리한다.

| 테이블 | 키 | 내용 |
|---|---|---|
| `episodes` | `episode_id` | 시뮬레이터 버전, config hash, seed, `plant_profile_id`, 운전모드, 길이, 샘플링주기 |
| `profiles` | `plant_profile_id` | domain-randomization 파라미터와 profile checksum |
| `signals` | `episode_id`, `time_s` | 배포 시점에 관측 가능한 센서·명령. v2 기본 계약에서는 사전 계산 알람 열을 포함하지 않음 |
| `oracle_labels` | `episode_id`, `time_s` | D1의 현재 참상태, 고장 주입정보, 안전상태, 열처리 지수와 fouling oracle. D3는 별도 schema에서 health/EOL/RUL/censor 라벨을 구현 |

현재 의존성 없는 참조 구현은 결정적 CSV를 만들고 `dataset_manifest.json`, `dataset_schema.json`, 생성 config, 소스 체크섬, episode split manifest를 함께 고정한다. 대용량 `D2`부터는 동일 논리 스키마의 Parquet 변환을 별도 버전으로 추가한다. 라벨 생성 후 `signals`와 `oracle_labels`의 컬럼 교집합은 조인 키 3개 외 0개여야 한다.

## 4. 합성 데이터 생성 계획

### 4.1 에피소드 단위

- 기본 길이: 1,800초, `dt=0.5초`인 3,600개 시점
- 장기 열화/RUL: 3,600~14,400초. 시간가속계수를 메타데이터에 기록
- 기동 안정화 구간: 최소 300초
- 고장 시작시각: 기동 완료 뒤 전체 길이의 25~75%에서 균등 표본
- 고장 지속시간: 순간고장이 아니라면 15~300초 또는 에피소드 종료까지
- 난수원: v2 모델의 공정·센서 잡음용 `noise_seed`와 profile·고장시작·지속시간·심각도 표본용 `sampling_seed`를 분리 기록한다. 정상/고장 counterfactual pair는 같은 `noise_seed`를 공유한다.
- 한 에피소드의 모든 파생 window는 같은 split에만 배정
- 각 episode/run은 새 simulator 상태에서 독립 초기화. 동일 profile/noise의 counterfactual pair도 상태 객체를 공유하지 않음

고장 주입시각, 참 물리상태의 최초 변화, 허용 센서에서 영향이 처음 관찰되는 시각을 각각 `fault_injection_time_s`, `physical_effect_time_s`, `fault_effect_time_s`로 기록한다. 탐지 적격 episode의 시간순서는 `fault_injection_time_s <= physical_effect_time_s <= fault_effect_time_s`여야 하며, 감사기가 `physical_effect <= observable effect`를 독립 검사한다. 관측효과가 episode 안에 도달하지 않으면 빈 시각과 `detection_eligible=0`을 명시해 임의의 고장 시작시각을 탐지 기준으로 대체하지 않는다. 급격한 고장은 주입 전 예측 대상이 아니라 **발생 후 신속 탐지** 대상으로 평가한다.

D1 episode와 `run_scenarios.py`의 run은 의도적으로 독립 초기화하며 campaign context를 feature로 쓰지 않는다. 생산→CIP→재기동 상태 연속성은 별도 `run_campaign.py`에 구현돼 있다. 고정 transition policy `1.0.0`은 물리 상태·RNG·FIFO·tank·HX·soil을 carry하고, 생산 tank를 CIP에서 격리하며, FDV를 순간 이동시키지 않고 divert 명령하고, phase 경계 PI integral을 reset한다. 재기동 chemical/product interface, 제품경계 조성과 제품온도가 적합해질 때까지 once-through drain한다. 저장된 complete/incomplete 예시는 chemical·dilution 부적합이 모두 0 L이고, incomplete-CIP만 surface-soil proxy로 인해 hygiene·통합 quality·unsafe 1,508.333333 L가 발생했다. 이 campaign 결과는 D1 분류나 D3 RUL 학습자료에 섞지 않는다.

### 4.2 plant profile domain randomization

동일한 명목 공장을 반복 복사하지 않도록, 에피소드보다 먼저 `plant_profile_id`를 만든다. 아래 범위는 실제 공장 규격이 아니라 v2 연구용 분포다.

| 항목 | 기준 대비 ID 학습 범위 | OOD 시험 예시 |
|---|---:|---:|
| 정격 유량 | 0.80~1.20배 | 0.65~0.79, 1.21~1.35배 |
| 원유 유입온도 | 2~10°C | 0~2, 10~14°C |
| 살균 설정온도 | 73~76°C | 학습에 없는 설정 조합 |
| 재생 효과도 | 0.82~0.95 | 0.70~0.81 |
| heater/regenerator 시정수 | 기본값의 0.70~1.50배 | 0.50~0.69, 1.51~2.00배 |
| 가열·냉각 최대효율 | 기본값의 0.80~1.20배 | 경계 밖 또는 결합 극단값 |
| PI 게인 | 기본값의 0.60~1.40배 | 학습에 없는 게인·지연 조합 |
| 센서 백색잡음 표준편차 | 0.02~0.20°C | 0.20~0.50°C, colored noise |
| FDV 응답지연 | 0.5~2.0초 | 2~5초 또는 불규칙 지연 |
| 압력·펌프 기준값 | v2 config의 ±10% | 경계 밖·압력역전 근접 조합 |
| fastest-flow 효율 | 0.80~0.90 | 경계 밖·RTD 교란 조합 |
| FDV 이동시간·후단 체류 | 이동 0.15~0.35초, 후단 3~7초 | 느린 밸브·긴 후단 inventory 조합 |
| balance tank | 용량 1,000~1,600 L, 초기량 400~800 L, 열손실 시정수 3,600~14,400초 | 저재고·긴 return 체류 조합 |
| 정지 유체 냉각 | 시정수 1,800~7,200초 | 장기 정전·정지 열이력 조합 |
| CIP 제거·수력 | caustic 0.004~0.012/s, acid 0.002~0.008/s, 기준속도 1.2~1.8 m/s, hold-up 80~150 L | 약한 세정·큰 hold-up 조합 |

물리적으로 연관된 파라미터는 독립 난수로 무작정 뽑지 않는다. 예를 들어 holding-tube volume, 유량, 명목 체류시간은 일관된 식으로 결합하고, 불가능한 열교환·압력 조합은 생성 단계에서 거부한다.

### 4.3 고장·운전모드 카탈로그

| 코드 | 시나리오 | v2 주입 형태 | 상태 | 주 ML 과제 |
|---|---|---|---|---|
| `N00` | 정상 생산 | 고장 없음 | 구현 | 정상 기준선 |
| `M01` | 기동/정지 | 표본화된 구간에 유량·가열·부스터 정지 | 구현 | mode 인식, 오경보 억제 |
| `M02` | 정상 CIP cycle | 고정 phase recipe | 구현 | 생산/CIP 구분 |
| `F01` | steam loss/degradation | 가열능력 step | 구현 | 신속 탐지·분류 |
| `F02` | flow surge | 실제 유량 step | 구현 | 체류시간 이탈 탐지 |
| `F03` | control sensor bias | 양의 bias step | 구현, `sensor_bias_high` 별칭 통합 | 센서 진단 |
| `F04` | safety sensor bias | 양의 bias step | 구현 | 센서 진단·안전경보 |
| `F05` | dual/common sensor bias | 두 센서 공통 양의 bias step | 구현 | 공통원인 OOD·안전 |
| `F06` | flowmeter bias + surge | 실제 surge + 표시유량 음의 bias step | 구현 | estimated residence 오류 탐지 |
| `F07` | booster pump failure | 펌프능력 step 감소 | 구현 | 압력차·누설 위험 |
| `F08` | regenerator leak/pressure inversion | 펌프능력 감소 + 누설 step | 구현 | 교차오염 위험 탐지 |
| `F09` | progressive fouling | 활성구간 fouling 증가율 가속 | 구현 | 건강도·RUL |
| `F10` | valve stuck forward + steam loss | 밸브 고착 + 가열능력 step 감소 | 구현 | 복합 안전고장 |
| `F11` | cooling utility loss | 냉각능력 step 감소 | 구현 | 출구온도 예측·경보 |
| `F12` | incomplete cleaning | caustic 활성구간의 세정효율 저하 | 구현, 정상 CIP counterfactual 사용 | 세정상태·잔류 fouling 탐지 |
| `F13` | power failure | 유량·가열·펌프 정지, fail-divert | 구현 | 전원상실 탐지·안전 회송 확인 |
| `F14` | sensor drift | control 온도센서 편향 ramp | 구현 | 완만한 센서 열화 탐지 |
| `F15` | sensor dropout | 계측값 hold-last-value | 구현 | 결측·동결 진단 |
| `F16` | slow valve | 저온 회송 요구 + FDV 이동시간 증가 | 구현 | 느린 밸브 응답 탐지 |
| `F17` | valve leakage | 저온 회송 요구 + 최소 forward opening | 구현 | 밸브 누설·부분개방 탐지 |

v2의 호환 별칭 `sensor_bias_high`는 `control_sensor_bias_high`와 같은 canonical label로만 매핑하고 별도 클래스로 중복 집계하지 않는다. 기본 taxonomy는 `SCENARIO_NAMES`의 레거시 별칭을 제외한 물리 동작을 정확히 한 번씩 포함하며, 생성기가 이 등가성을 실행 전에 검사한다. `F12`는 생산 정상파형이 아니라 동일 recipe·config·noise의 `cip_cycle`을 counterfactual로 사용하고, 고장구간을 실제 caustic circulation과 겹치게 결정적으로 표본화한다.

### 4.4 데이터 규모

| 데이터 버전 | 구성 | 예상 크기 | 목적 |
|---|---|---:|---|
| `D0-smoke` | 20 canonical class × 1회 × 1,800초 / 0.5초 | 72,000행 | 스키마·결정성·label audit만 |
| `D1-pilot` | 12 profiles × 20 canonical class × 5 seeds × 1,800초 | 1,200 episodes, 약 432만 행 | 파이프라인·베이스라인 개발 |
| `D2-ood-dev` | ID 12 + OOD 4 profiles × 20 classes × 3 replicates × 900초 | 960 episodes, 172.8만 행 | 검증된 synthetic profile-support-shift 실행 개발 |
| `D2-benchmark` | 80 profiles × 20 canonical class × 최소 5 seeds | 최소 8,000 episodes, 약 2,880만 행 | 공식 ID/OOD 비교 |
| `D3-RUL` | 80 profiles × profile당 10 trajectories, acceleration/censor 표본 | 800 장기에피소드 | 구현된 건강도·RUL 파이프라인의 정규 규모 |
| `D4-control` | 초기상태·외란·profile 조합 최소 10,000 episodes | rollout 기반 | 제어 최적화 독립 평가 |

구현된 canonical class는 `N00`, `M01`, `M02`, `F01`~`F17`의 20개다. simulator 호환 별칭 `sensor_bias_high`는 `F03`과 중복 집계하지 않는다.

## 5. 라벨 정의

### 5.1 운전상태·이상·고장 라벨

| 라벨 | 정의 |
|---|---|
| `plant_mode` | `signals.csv`의 context. `STARTUP`, `PRODUCTION`, `RECOVERY`, `DIVERT`, `CIP_*` 상태이며 기본 feature allowlist에는 포함하지 않음 |
| `fault_active` | 주입된 고장이 현재 물리상태에 영향을 주는 구간. 기동·정상 회송·정상 CIP는 0 |
| `behavior_active` | 고장뿐 아니라 M01/M02까지 포함한 해당 canonical 동작의 활성구간 |
| `canonical_code` | `oracle_labels.csv`와 `episodes.csv`의 `N00/Mxx/Fxx` 단일 taxonomy. 현재 복합 시나리오는 하나의 canonical code이며 별도 multi-hot 열은 아직 없음 |
| `fault_effect_time_s` | 관측가능 상태가 정상 counterfactual에서 최초로 유의하게 벗어난 시각 |
| `physical_effect_time_s` | 참 물리상태가 mode-matched counterfactual에서 최초로 달라진 시각. 관측효과보다 늦을 수 없음 |
| `safety_event` | 참온도·참체류시간·열처리 proxy 중 하나가 내부 기준 미달이거나 압력역전/누설 위험이 참인 상태 |
| `unsafe_forward_event` | post-FDV FIFO를 거쳐 **제품 경계에 도달한** 비안전 부피 `unsafe_forward_l > 0`. FDV 명령이나 센서 위치의 `safety_event`와 같은 뜻이 아님 |
| `diversion_event` | 정상 기동 이후 FDV가 divert로 전환된 사건 |
| `cooling_excursion` | forward 제품의 참 출구온도가 내부 상한을 넘는 사건 |

point label과 event label을 모두 보존한다. 30초 동안 지속된 한 고장을 60개의 독립 정답으로 세지 않고, event recall은 사건 하나당 한 번만 센다.

### 5.2 조기경보 라벨

관측시각 `t`에서 다음 horizon 안에 사건이 처음 발생하면 양성이다.

`y_event@H(t) = 1  if  t < t_event <= t + H`

- horizon: 5, 15, 30, 60, 120초
- 현재 생성되는 대상 4종: `safety_event`, `unsafe_forward_event`, `diversion_event`, `cooling_excursion`
- episode 종료까지 horizon 전체를 관측할 수 없는 행은 0으로 간주하지 않고 빈 값으로 right-censor한다.
- 사건 이후 recovery가 끝날 때까지의 window는 새 조기경보 표본으로 만들지 않는다.
- abrupt failure는 사전 징후가 없으므로 0초 전 조기경보 성능을 주장하지 않는다. `fault_effect_time_s` 이후 detection latency로 평가한다.

### 5.3 soft sensor 목표

| 목표 | 입력에서 반드시 제거할 값 | 평가 단위 |
|---|---|---|
| 참 holding outlet 온도 | 두 센서의 참 bias, 참 holding 온도, `actual_safe` | °C |
| 참 체류시간 | 참유량, holding volume로 직접 계산한 값, `actual_safe` | 초 |
| 참 product outlet 온도 | 참 cooling factor, 참 product 온도 | °C |
| 참 differential pressure/leak risk | 참 압력차, leak flag와 직접 파생값 | bar/확률 |
| `fouling_index` | 주입 fouling factor, regeneration/heater capacity factor | 0~1 |

동일 방정식의 입력을 그대로 제공해 정답을 재계산하는 것은 ML soft sensor가 아니라 물리식 oracle이다. 이 oracle은 성능 상한 기준선으로만 별도 보고한다.

### 5.4 건강도와 RUL — 구현된 독립 `D3-RUL`

D1은 `fouling_index`를 oracle로만 보존하고 health/EOL/RUL을 만들지 않는다. 별도 D3 파이프라인은 **중단 없는 단일 생산 run의 scalar fouling**에 대해 `health_index = 1 - fouling_index`, EOL evidence, exact RUL과 right-censor lower bound를 생성한다. 생산→CIP→재기동 campaign이나 반복 정비 후 수명회복은 D3 범위 밖이다.

가속계수는 fouling 성장률에만 적용한다. `time_sim_s`와 관측 feature `operating_time_meter_h = time_sim_s / 3600`은 가속되지 않은 simulator clock이고, equivalent time은 `simulation time × acceleration factor`로 oracle에만 기록한다. 최소 monitor 시간과 정상 forward persistence로 arm된 뒤 아래 조건 중 먼저 완료된 원인을 EOL로 고정한다.

1. `fouling_index >= 0.80`
2. steam valve 90% 이상이 60 simulator seconds 지속
3. matched clean-reference 대비 300초 rolling energy/L 비율이 1.25 이상이고 그 상태가 300초 지속
4. 정상 forward arm 뒤 forward 상실이 5초 지속

EOL state machine은 내보낸 sample grid에서만 평가하며 threshold 시작·완료시각과 첫 원인을 `eol_evidence.csv`에 남긴다. 관측 EOL에만 `RUL(t) = max(0, t_EOL - t)`를 채운다. EOL 전에 administrative horizon에 도달한 trajectory는 exact RUL을 빈 값으로 두고 `time_to_event_or_censor_*`와 `rul_lower_bound_*`만 기록한다. 종료시각을 pseudo-failure나 RUL 0으로 대체하지 않는다.

저장된 `D3-RUL-v2.2-smoke`는 6 profiles × 4 trajectories = 24 trajectories(그중 ID 범위와 겹치지 않는 OOD profile 1개), signal/label 각 990행, EOL 13건, right-censor 11건이다. Profile 단위 60/20/20+OOD split과 EOL/censor 재구성 감사 26/26을 통과했고, train-only 건강도/RUL 기준선은 175개 RUL landmark를 냈다. 이는 파이프라인 재현 증거이지 현장 또는 OOD 성능 benchmark가 아니다. 상세 계약은 [D3_RUL.md](D3_RUL.md)에 있다.

### 5.5 v3 반복 정비/CIP lifecycle — 구현된 시뮬레이션, ML 성능 미평가

`lifecycle.py`는 D3와 독립적으로 4개 asset의 가역 fouling, 비가역 damage, virtual age, complete/incomplete CIP, scheduled calibration/minor repair/overhaul, replacement, Weibull/damage/fouling competing event와 right-censored service interval을 생성한다. 기본 36-cycle trace는 `run_digital_twin.py` 산출물에 포함된다.

v3 lifecycle을 이용한 ML 실험 계약은 다음과 같다.

| ID | 목표 | 금지 누수 | 평가 |
|---|---|---|---|
| `EXP-V3-101` | raw/corrected/uncertainty 조합별 교정 인지 고장탐지 | true/filtered sensor value, guard 결과 직접 사용 | macro-F1, false alarms/hour, delay |
| `EXP-V3-102` | 12개 PLC cause 분류 | PLC cause/output을 입력으로 쓴 결과와 쓰지 않은 결과 혼합 금지 | cause recall, nuisance alarm |
| `EXP-V3-103` | 5개 HACCP 공정 gate 이탈 조기경보 | evidence status/hash/lot outcome 입력 금지 | horizon AUPRC, lead time, false-safe |
| `EXP-V3-104` | multi-cycle component health 추정 | damage/fouling/virtual-age oracle 입력 금지 | MAE, Spearman, calibration |
| `EXP-V3-105` | competing-risk RUL | censored interval의 exact RUL 대체 금지 | C-index, integrated Brier, coverage |
| `EXP-V3-106` | 정비정책 counterfactual 비교 | 다른 random stream의 정책간 단순 비교 금지 | availability, failure, maintenance-cost proxy |
| `EXP-V3-107` | P&ID graph/sensor ablation | 같은 asset/service interval의 split 교차 금지 | worst-cause recall, missing-sensor robustness |

service interval 전체와 같은 asset/config/seed family는 한 split에만 둔다. scaling·threshold·imputation은 train에서만 적합하고, 마지막 administrative interval은 right censor로 유지한다. 이 표는 실험 구성만 완료된 상태이며 학습 성능이나 정비효과를 주장하지 않는다.

## 6. 분할과 누수 방지

### 6.1 공식 split

`D2-benchmark`의 profile을 생성 전에 고정한다.

아래 80-profile 표는 여전히 prospective external-confirmatory 후보 설계이며, 현재 생성된 `D2-ood-dev`와 같은 데이터셋이 아니다. `D2-ood-dev`는 train-ID 7 profiles/420 episodes, validation-ID 2/120, test-ID 3/180, test-OOD-profile 4/240의 작은 개발 set이다.

| split | profile | 용도 |
|---|---:|---|
| train-ID | 48 | 모델 적합 |
| validation-ID | 12 | 하이퍼파라미터·threshold·calibration |
| test-ID | 12 | 최종 ID 평가, 1회 개봉 원칙 |
| test-OOD-profile | 8 | 학습범위 밖 설비·운전 파라미터 |

추가로 `test-OOD-fault`에는 학습에서 제외한 고장 종류 1~2개, `test-OOD-combination`에는 학습하지 않은 복합고장 조합, `test-OOD-sensor`에는 dropout·stuck·colored noise를 둔다. OOD 세트로 threshold나 하이퍼파라미터를 조정하지 않는다.

### 6.2 누수 방지 체크리스트

- [x] profile 단위의 결정적 60/20/20 split을 만들고 row 단위 무작위 분할을 금지한다.
- [x] 같은 `plant_profile_id`, config hash, seed family, counterfactual pair는 한 split에만 둔다.
- [x] window manifest가 제공되면 같은 episode의 overlapping window가 다른 split에 들어가는지 감사한다.
- [ ] window가 고장 주입 이후 데이터를 과거 입력처럼 포함하지 않는지 timestamp로 검사한다.
- [x] onset을 무작위화하고 절대 `time_s`는 입력 allowlist에서 제외한다.
- [ ] scaling, imputation, feature selection, PCA는 train만 fit한다.
- [x] `scenario`, modifier, 참상태, 미래 알람, 결과 누적량을 `signals.csv`에서 제거한다.
- [ ] 목표의 직접 파생변수는 feature-lineage 표에서 차단한다.
- [x] 정상 counterfactual과 고장 episode를 같은 `counterfactual_group_id`로 만들고 같은 split에 둔다.
- [ ] test-ID와 모든 OOD test는 최종 모델 고정 전 열지 않는다.
- [ ] window 수가 아니라 profile/episode/event 단위로 bootstrap과 유의성 검정을 수행한다.

## 7. 공통 전처리와 통계 프로토콜

- 원시 0.5초 신호를 보존하고 1초·2초·5초 downsample은 ablation으로 비교한다.
- 기본 입력 window는 10, 30, 60, 120초, stride는 train 2.5초, test 0.5초다.
- 온도·압력·유량의 1차 차분, rolling mean/std/slope, 센서간 차이는 train pipeline 안에서 계산한다.
- 결측과 sensor dropout은 0으로 채우기 전에 missingness mask와 time-since-last-observation을 함께 준다.
- 표준화는 profile별 전체 통계를 쓰지 않고 train 전역 또는 과거 시점의 running 통계만 쓴다.
- class weighting, focal loss, balanced sampler를 비교하되 자연 발생률을 보존한 test에서 평가한다.
- 일반 후속 모델은 최소 5개 학습 seed를 권고하되, 이미 동결된 FlowTwin v0.2 matrix는 예외적으로 등록 3 seeds를 정확히 사용한다.
- 주 결과는 seed 평균과 95% profile-cluster bootstrap CI를 보고한다.
- 두 모델 비교는 동일 episode의 paired 결과로 하고, 평균 timestamp 정확도로 유의성을 부풀리지 않는다.
- threshold는 validation에서 고정한 뒤 test에서 변경하지 않는다.

### 7.1 FlowTwin benchmark v0.2 고정 프로토콜

위 원칙은 전체 ML roadmap의 넓은 권고이고, 현재 구현된 FlowTwin benchmark의 정확한 계약은 `flowtwin_benchmark_contract.json` `0.2.0`이다.

- 등록 행렬: FlowTwin-Guard 1 + neural baseline 6 + ablation 9 = **16 variants**; seeds `20260727`, `20260728`, `20260729`.
- 여섯 baseline: `tcn`, `causal_transformer`, `static_pid_gnn`, `dynamic_gnn`, `twin_residual_tcn`, `dspr_diagnostic_adaptation`.
- DSPR은 physics-residual dual stream, physics-guided dynamic graph, adaptive causal window를 담은 가장 가까운 선행연구 비교군이다. 저자 구현이 공개되지 않은 상태에서 forecasting 식을 six-output causal diagnosis로 독립 변환한 **diagnostic adaptation**이며 exact author-code reproduction이 아니다.
- 모든 supervised/transport/counterfactual loss와 exhaustive evaluation은 `valid_mask & eval_mask`를 쓴다. `valid_mask`는 causal context/padding, `eval_mask`는 중첩 window에서 한 row의 owning loss region이다.
- Event-aware sampler의 onset/middle/end landmark는 반드시 그 window의 owning `eval_mask` loss region에 속한다. Non-`N00` target-bearing train episode가 하나라도 target row를 loss에 남기지 못하면 preflight가 실행을 거부한다.
- Conformal calibration은 validation-ID의 **counterfactual group × observable production/CIP mode** block별 true-class nonconformity/ID energy 최대값을 쓴다. D1 validation은 profile 2개이므로 `COV`는 empirical synthetic row coverage로만 보고하며 profile shift coverage guarantee는 금지한다.
- 추론 단위는 `plant_profile_id`, seed는 technical repeat이다. `test_id` 20-class macro-F1, combined-test event F1/recall, false-alarm onset/negative-hour, unsafe-forward L를 보고한다.
- 등록 latency endpoint는 miss에 observable effect에서 event 종료까지의 남은 horizon을 부여한 **mean horizon-penalized latency**이며 event recall·miss count와 항상 같이 보고한다.
- Tier는 `development`, `pilot`, `protocol_complete_synthetic`이다. 최상위 label조차 합성 ID/OOD 프로토콜 완료만을 뜻하며, runner는 external-confirmatory·현장·HACCP 판정을 절대 부여하지 않는다.

### 7.2 FlowTwin v0.3 post-hoc 개발 후보 계약

`flowtwin_hybrid_v03_dev`는 `flowtwin_v03_candidate_contract.json` `0.1.0`으로 관리하는 별도 후보다. D2 test-ID/OOD를 후보 동결 전에 이미 확인했으므로 `d2_test_seen_before_freeze=true`, `status=development_only`이며, 후보가 포함된 run은 조건과 무관하게 `development`다. 이 row는 v0.2의 **16 variants registry를 변경하지 않는다**. `all`·`models`·`ablations` alias도 후보를 포함하지 않고, 후보의 전체 ID를 명시해야만 선택된다.

후보 구조는 causal TCN과 persistence-skip nominal observer residual을 병렬로 만들고, 관측 route와 profile별 volume/실측 flow의 `V/Q` delay를 쓰는 graph branch를 통과시킨 뒤 pointwise sigmoid convex gate로 융합한다. `TargetContract.code_is_fault` mask에서 fault/non-fault class partition을 직렬화하고 binary fault probability와 두 conditional class distribution을 조합하므로 `sum(P(fault classes)) = P(anomaly)`가 정확히 유지된다. OOD score는 `log1p(mean normalized observer residual²) + log1p(route-active transport error)`인 physical nonconformity이며 v0.2 energy와 수치 비교하지 않는다.

W96은 D2를 본 뒤 선택한 post-hoc 설계값이다. Train-only critical-delay gate는 `valid_mask & eval_mask & flow>1 & observable route gate>0` 행에서 `local index - V/Q/dt >= 0`인 비율을 edge별로 계산한다.

| exact-FIFO edge | exhaustive coverage (valid/eligible) | sampled-train coverage (valid/eligible) |
|---|---:|---:|
| `L-005` | 0.9783254115 (695,290/710,694) | 0.9092718898 (154,378/169,782) |
| `L-007` | 0.9980075813 (709,278/710,694) | 0.9916598933 (168,366/169,782) |
| `L-008` | 0.9982100432 (461,753/462,581) | 0.9921487564 (104,633/105,461) |
| `L-009` | 0.9979765706 (461,645/462,581) | 0.9911246812 (104,525/105,461) |
| `L-010` | 0.9980154827 (461,663/462,581) | 0.9912953604 (104,543/105,461) |

판정은 pooled 값이 아니라 edge별 최솟값을 쓰며 exhaustive `≥0.95`, sampled `≥0.90`을 요구한다. 감사 중 연 split은 `train_id`뿐이고 non-train row는 0개다. W64의 사후 확인 최소값은 각각 `0.851460685`, `0.786727010`이었고 W96 선택 근거와 함께 공개한다. 분모가 0인 edge 또는 기준 미달 edge가 있으면 학습 전에 실패한다.

확률 교정은 validation-ID에서만 binary-fault temperature와 fault/non-fault conditional temperature를 적합해 hierarchy를 보존한 뒤 기존 block-max conformal로 이어진다. Candidate contract의 같은 hysteresis·persistence·cooldown grid와 profile별 제약은 후보 run에서 선택한 **모든 variant**에 적용한다. 각 variant는 자기 validation score에서 운전점을 선택하므로 선택 config는 다를 수 있으나 탐색공간·목표·제약은 같다. 운전점이 없으면 `no_operating_point`로 test iterator 생성 전에 실패하며 point-threshold fallback은 금지한다.

기존 point-threshold 결과는 `event_detection`·`false_alarms`·`unsafe_forward_l_before_first_post_effect_alarm`에 보존하고, 상태기계 경보는 별도 `operational_*` endpoint로 보고한다. 따라서 후처리가 기존 point-threshold 지표를 대체하거나 숨기지 않는다. 확정된 개발 결과에서 TCN은 operational grid `0/75`로 fail-closed됐고, Hybrid은 ID macro-F1은 가장 높았지만 전체·OOD·operational 전반에서 우월하지 않았다. D2 결과는 synthetic post-hoc 개발 증거이며 성능 우월성·external-OOD·confirmatory·현장·살균·HACCP·제품안전·출하권한 근거가 될 수 없다. 논문 가설 판정에는 설계에 쓰이지 않은 새 profile set의 독립 봉인 평가가 필요하다.

## 8. 기준선과 모델군

### 8.1 반드시 포함할 비학습·고전 기준선

- 현행 안전 규칙: 독립 안전온도, 최소 추정 체류시간, 압력차, FDV 상태
- 마지막 값 유지와 계절성 없는 persistence forecast
- 이동평균, EWMA, CUSUM, Shewhart limit
- 센서 disagreement 고정 threshold
- 선형/로지스틱 회귀, ridge/lasso
- ARX 또는 저차 상태공간모델과 Kalman filter
- PCA reconstruction error, One-Class SVM, Isolation Forest
- 수작업 PI와 보수적 PID tuning

규칙 기반 안전 로직은 단순히 이겨야 할 약한 baseline이 아니라, ML이 침범하지 않아야 하는 안전 상한선이다.

구현된 rule·EWMA·CUSUM 기준선은 `plant_mode`를 기본 feature로 쓰거나 mode 분기에 사용하지 않고, 관측/제어 relay인 `cip_cycle_active`로 생산 안전규칙과 CIP response 규칙을 gating한다. 전원과 온도센서 상태도 oracle 필드 자체가 아니라 관측 interface `power_good_signal`/`temperature_sensor_quality_ok`를 입력하지만, 현재 합성구현에서 두 relay가 truth를 결정적으로 mirror한다는 제약을 함께 보고한다.

### 8.2 비교할 ML 모델군

| 과제 | 경량 모델 | 시계열/딥러닝 | 불확실성 후보 |
|---|---|---|---|
| fault detection/classification | Random Forest, Extra Trees, gradient boosting | 1D-CNN, TCN, GRU/LSTM, time-series Transformer | deep ensemble, MC dropout |
| unsupervised anomaly | Isolation Forest, PCA, robust covariance | autoencoder, VAE, forecasting residual model | ensemble residual, conformal score |
| soft sensor/forecast | ridge, Random Forest, gradient boosting, Kalman filter | TCN, GRU, Transformer, neural state-space | quantile regression, conformal interval |
| health/RUL | Cox/Weibull AFT, survival forest, gradient boosting | TCN/GRU survival, DeepHit 계열 | censored conformal/ensemble interval |
| control surrogate | linear state-space, Gaussian Process | neural state-space, latent dynamics | ensemble dynamics |
| policy optimization | grid/random search, Bayesian optimization, MPC | constrained MPC, offline RL, safe RL | robust/ensemble MPC |

모델 복잡도는 경량 기준선을 유의하게 이길 때만 늘린다. 실시간 후보는 추론 지연, 메모리, 입력 센서 수를 함께 보고한다.

## 9. 지표

### 9.1 이상·고장·조기경보

- event-level recall, precision, F1
- macro-F1와 class별 recall
- AUPRC 우선, AUROC 보조
- 정상 생산시간당 false alarms와 episode당 nuisance alarm 수
- detection delay: `첫 지속 경보 - fault_effect_time_s`
- 조기경보 lead time: `event time - 첫 유효 경보`
- time-to-detect 분포의 median, p90
- unsafe product volume before first alarm
- start/stop/CIP 구간의 mode별 false alarm rate

경보는 3초 중 2초 이상 threshold를 넘는 등 persistence rule을 적용한 최종 이벤트로 평가하며, point probability와 혼동하지 않는다.

FlowTwin v0.2 등록 latency는 성공한 event의 delay 분포만으로 판정하지 않고, miss에 남은 observable-event horizon을 부여한 mean horizon-penalized latency를 사용한다. 성공-event median/p90은 보조 기술통계이며 event recall·miss count없이 단독 인용하지 않는다.

### 9.2 회귀·soft sensor·forecast

- MAE, RMSE, bias, p95 absolute error
- operating range로 정규화한 NRMSE
- 임계값 주변(온도 72°C 부근, 체류 15초 부근)의 별도 오차
- interval coverage와 mean interval width
- `actual_safe` 판정으로 변환했을 때의 false-safe rate와 false-divert rate

### 9.3 건강도·RUL

- health index MAE와 Spearman correlation
- RUL MAE/RMSE, full life 대비 normalized MAE
- ±10%, ±20% 허용범위 내 비율
- censored 데이터를 포함한 concordance index와 integrated Brier score
- 90% prediction interval coverage/width
- 조기정비 비용, 고장 후 정비 비용, 불필요 정비 횟수의 시뮬레이션 비용

### 9.4 제어

- `unsafe_forward_l`와 safety-interlock violation 수
- forward 제품의 최소 참온도·최소 참체류·최소 열처리 proxy
- thermal energy kWh/forward kL
- 정상 기동 제외 diversion fraction
- 설정값 IAE/ISE, overshoot, settling time
- steam valve total variation과 saturation 시간
- maximum product outlet temperature
- profile·외란별 worst-case와 p95, 평균값

## 10. 세부 실험 카드

아래 성능 카드는 정규 규모 실험 기준으로 `예정`이다. `EXP-HTST-ML-000`/`001`의 D1 실행경로와 `EXP-030`/`031`에 필요한 D3 생성·감사·train-only 기준선 경로는 구현돼 축소 smoke로 확인했지만, 정규 데이터의 합격 기준을 통과했다고 주장하지 않는다.

### EXP-HTST-ML-000 — 데이터·라벨 무결성

- 목적: ML 전에 시뮬레이터가 결정적이고 물리·라벨·split 불변식을 만족하는지 확인
- 데이터: `D0-smoke`, 이후 `D1-pilot`
- 검사: 동일 seed byte-level 재현, 다른 seed의 잡음 차이, 질량/에너지/holding queue 보존, 신호 범위, NaN/inf, monotonic timestamp, fault onset 무작위성, 조인 키를 제외한 oracle-feature 교집합 0개
- 합격: 필수 불변식 100%, 동일 seed checksum 일치, episode/split 중복 0개, forbidden feature 0개
- 산출물: `dataset_manifest.json`, schema report, split audit, label prevalence table

### EXP-HTST-ML-001 — 규칙·통계 기준선

- 목적: 현행 임계값, sensor disagreement, EWMA/CUSUM으로 얻을 수 있는 성능 상한과 오경보 기준 설정
- 데이터: `D1-pilot`, `S0`~`S2`
- 라벨: `fault_active`, canonical fault code, 각 safety event
- 지표: event recall, false alarms/hour, detection delay, unsafe volume before alarm
- 합격: 구현·재현 완료가 gate. 이후 ML은 동일 false-alarm 예산에서 이 결과와 paired 비교

### EXP-HTST-ML-010 — 정상-only 이상 탐지

- 목적: 학습에 없던 고장과 복합고장을 fault label 없이 탐지
- train: 정상 생산·정상 기동·정상 CIP만. validation에 알려진 소수 고장, test에 미지 고장 포함
- 모델: PCA, Isolation Forest, One-Class SVM, forecasting residual, autoencoder
- 누수 방지: 고장 episode 및 같은 counterfactual profile을 normal train에 넣지 않음
- 합격: test-ID event recall ≥0.95 at ≤1 false alarm/운전 1시간, p90 detection delay ≤10초; OOD event recall ≥0.80

### EXP-HTST-ML-011 — 고장 분류·다중라벨 진단

- 목적: 현재 구현된 `F01`~`F17` 원인 분류와 복합고장 구성요소 식별
- 입력: 30/60초 window, `S0`, `S1`, `S2` 단계 비교
- 모델: logistic/RF/gradient boosting, 1D-CNN, TCN, GRU, Transformer
- 지표: macro-F1, class recall, multi-label mAP, confusion matrix, detection delay
- 합격: test-ID macro-F1 ≥0.90, 모든 안전관련 class recall ≥0.95, 정상 false positive ≤1/hour

### EXP-HTST-ML-012 — CCP·회송·냉각 조기경보

- 목적: 5/15/30/60/120초 내 safety/diversion/cooling event 예측
- 제외: 사전징후가 없는 abrupt fault의 발생 전 성능은 별도 표기하고 종합점수에서 제외
- 모델: logistic/gradient boosting, TCN/GRU/Transformer, survival/hazard model
- 지표: horizon별 AUPRC, recall at fixed precision, lead time, false alarms/hour
- 합격: progressive/observable-precursor 사건에서 30초 horizon AUPRC ≥0.85, recall ≥0.90 at precision ≥0.80, median lead time ≥15초

### EXP-HTST-ML-013 — 센서·설비 고장 위치분리

- 목적: control sensor, safety sensor, common bias, flowmeter, booster pump, regenerator를 구분
- 핵심 비교: 단일센서 `S0-control-only` 대 독립센서 `S0-core`, 압력 제외 대 포함
- 지표: source localization accuracy, class recall, common-bias miss rate
- 합격: 단일 센서 bias recall ≥0.95, dual/common bias recall ≥0.90, pressure inversion recall ≥0.98

### EXP-HTST-ML-020 — 온도·체류시간 soft sensor

- 목적: 센서 고장/결측 시 참 holding 온도와 체류시간 추정. 바이오 또는 lethality target은 제외
- 입력: 목표 직접 파생변수를 제거한 `S0/S1/S2`, 60초 history
- 모델: 물리식 oracle(상한), ridge/Kalman/gradient boosting, TCN/GRU/neural state-space
- 지표: MAE/p95, 임계값 주변 error, 90% interval coverage, false-safe rate
- 합격: ID holding temperature MAE ≤0.20°C·p95 ≤0.50°C, residence MAE ≤0.50초, 90% interval coverage 85~95%, 임계값 변환 false-safe rate ≤0.1%
- 안전조건: soft sensor는 독립 안전센서나 FDV 인터록을 대체하지 않고 진단·shadow mode로만 평가

### EXP-HTST-ML-021 — 다변량 상태 예측

- 목적: 5/15/30/60초 뒤 온도, 압력, 유량, 제품 출구온도를 예측해 제어와 조기경보에 공급
- 기준선: persistence, ARX/Kalman, linear state-space
- 모델: gradient boosting direct forecast, TCN/GRU/Transformer, neural state-space
- 지표: horizon별 MAE, transient 구간 MAE, interval coverage
- 합격: 모든 horizon에서 persistence 대비 RMSE ≥20% 감소, 30초 holding temp MAE ≤0.35°C, 90% interval coverage 85~95%

### EXP-HTST-ML-030 — 오염 건강도·예지보전

- 목적: 정상 제어가 온도를 유지하는 동안 숨어 있는 열교환기 오염도를 추정
- 데이터: `D3-RUL`은 중단 없는 단일 생산열화와 profile별 부하 차이를 담당한다. v3 lifecycle은 반복 CIP/정비 service interval을 별도로 구현했지만 아직 ML benchmark로 생성·분할·학습하지 않았다.
- 라벨: `fouling_index`, EOL condition, 정비필요 여부
- 구현 기준선: constant/ridge health. 정규 비교 후보는 EWMA energy residual, gradient boosting, state-space, TCN/GRU
- 지표: health-index MAE/Spearman, EOL 경보 lead, 불필요 정비율
- 합격: `fouling_index` MAE ≤0.05, Spearman ≥0.90, EOL 10분 전 recall ≥0.90 at precision ≥0.80

### EXP-HTST-ML-031 — RUL·censoring

- 목적: 서로 다른 오염속도와 부하에서 EOL까지 남은 시간 및 불확실성 예측
- 분할: degradation trajectory/plant profile 전체를 그룹 분할; 한 수명곡선의 앞부분과 뒷부분을 다른 split에 두지 않음
- 구현 기준선: train-event naive age, train-only Kaplan–Meier conditional RUL, ridge-health slope+KM fallback. 정규 비교 후보는 Weibull/Cox AFT, survival forest, boosting survival, GRU/TCN survival
- 지표: normalized MAE, concordance index, integrated Brier, interval coverage
- 합격: normalized RUL MAE ≤10%, C-index ≥0.85, ±20% 적중률 ≥0.90, 90% interval coverage 85~95%

### EXP-HTST-ML-040 — OOD·미지 고장 탐지

- 목적: profile, severity, fault class, 복합고장, 센서 noise가 학습범위 밖인지 식별
- test: `OOD-profile`, `OOD-fault`, `OOD-combination`, `OOD-sensor`를 각각 분리 보고
- 모델: Mahalanobis/ensemble disagreement/energy score/deep ensemble/conformal score
- 지표: OOD AUROC/AUPRC/FPR95, OOD에서 안전사건 recall, confidence-error correlation
- 합격: OOD AUROC ≥0.90, FPR95 ≤0.20, OOD 안전사건 recall ≥0.90; 불충족 시 자동 rule-only fallback

### EXP-HTST-ML-041 — domain randomization과 sim-to-sim 일반화

- 목적: 좁은 단일 config 학습보다 파라미터 무작위화가 미지 profile 성능을 높이는지 검증
- 비교: nominal-only, noise-only, parameter-only, full randomization, curriculum randomization
- 지표: ID/OOD 성능, calibration, worst-profile recall
- 합격: full randomization이 nominal-only 대비 OOD macro-F1 ≥10%p 개선하면서 ID macro-F1 하락 ≤2%p

### EXP-HTST-ML-050 — 확률 보정·선택적 예측

- 목적: 모델이 모를 때 확신하지 않고 abstain/fallback 할 수 있게 함
- 방법: Platt/isotonic/temperature scaling, deep ensemble, conformal prediction
- 지표: ECE, Brier score, risk-coverage curve, interval coverage
- 합격: ID ECE ≤0.03, OOD ECE ≤0.10, 10% abstention에서 안전관련 error ≥50% 감소, coverage gate 충족

### EXP-HTST-ML-060 — PI gain·setpoint 운영 최적화

- 목적: 기존 PI 구조를 유지한 채 gain, 비안전 범위 내 setpoint, anti-windup을 튜닝
- 방법: grid/random search → Bayesian optimization; ML 없이 얻는 제어 최적화 기준선
- 평가: `D4-control`의 unseen profile/외란 10,000 rollout
- 목적함수: 안전위반을 hard constraint로 먼저 걸고, 이후 energy/L → 정상 diversion → IAE → valve variation 순으로 lexicographic 최적화
- 합격: 10,000 rollout에서 unsafe forward 0 L, 냉각 상한 위반 0회, 기존 PI 대비 energy/forward kL ≥5% 감소, diversion 악화 ≤1%p

### EXP-HTST-ML-061 — 동역학 surrogate + constrained MPC

- 목적: `EXP-021` 예측모델로 미래 온도·압력·유량을 보고 steam/flow를 제한범위에서 최적화
- 비교: 기존 PI, 튜닝 PI, 선형 MPC, ensemble neural MPC
- hard constraint: FDV·압력차·최소체류 인터록은 별도 결정론 계층; ML이 override 불가
- 지표: 안전위반, energy/L, IAE, saturation, compute latency
- 합격: 튜닝 PI와 동일하게 unsafe 0 L, worst-profile safety violation 0회, energy ≥3% 추가 절감 또는 diversion ≥10% 상대 감소, 0.5초 제어주기 내 p99 추론 완료

### EXP-HTST-ML-062 — safe offline RL/정책 최적화

- 목적: 복합 외란에서 제한된 steam/flow setpoint 조정을 학습하되, 시뮬레이터 연구로만 평가
- 선행조건: `EXP-061` 통과, 독립 safety shield 구현, reward hacking audit
- 비교: behavior cloning, conservative offline RL, constrained policy, robust MPC
- 금지: safety penalty를 유한 reward로만 처리, FDV 직접제어, test rollout을 학습 replay에 재사용
- 합격: 100,000 stress rollout에서 unsafe forward 0 L와 rule violation 0회, MPC 대비 Pareto 개선, OOD 감지 시 즉시 safe PI fallback

### EXP-HTST-ML-070 — 통합 shadow-mode 시험

- 목적: detector + soft sensor + uncertainty + fallback + control advisory 전체 경로 검증
- 시나리오: 정상, 기동/정지, CIP, 단일고장 전부, 미지 복합고장, 센서 dropout, profile OOD
- 지표: end-to-end false alarm, missed safety event, first-alarm latency, fallback 정확성, compute/memory
- 합격: 안전사건 missed event 0건/공식 stress suite, ≤1 nuisance alarm/운전 1시간, OOD/fault 시 rule-only fallback 100%, 모든 결정과 입력 snapshot audit log 보존

## 11. Ablation 계획

각 실험은 최소한 관련 항목을 한 번씩 제거해 성능 기여를 확인한다.

| 축 | 비교 |
|---|---|
| 독립 센서 | control만 / safety만 / 두 센서 / disagreement 포함 |
| 압력 계측 | 압력 없음 / 두 압력 / differential + pump speed |
| PLC 파생값 | raw only / estimated residence·sensor disagreement 포함. `alarm_count`와 `alarm_unsafe_forward`는 모든 조건에서 금지 |
| window | 10 / 30 / 60 / 120초 |
| sampling | 0.5 / 1 / 2 / 5초 |
| 물리 feature | 없음 / 물리 residual / hard physics constraint |
| domain randomization | 없음 / noise만 / parameter만 / 전체 |
| 고장 강도 | 명확한 고장만 / 약한 고장 포함 / 경계값 주변 |
| 불균형 처리 | 없음 / class weight / focal loss / balanced sampling |
| mode context | mode 없음 / known mode / mode를 먼저 예측 |
| uncertainty | 단일모델 / ensemble / conformal |
| 결측 대응 | 단순 보간 / mask / learned imputation |
| direct-status relay | `power_good_signal`·`temperature_sensor_quality_ok` 포함 status-aware / 두 relay를 제외한 sensor-only |
| 절대시간 | 포함 / 제외; 포함 모델이 onset을 외우는지 확인 |
| 학습 데이터량 | 10 / 25 / 50 / 100% profile learning curve |

ablation은 한 번에 한 축을 바꾸고 동일 split·seed를 사용한다. 평균 개선뿐 아니라 safety-related recall의 최악 profile 값과 95% CI를 함께 본다.

## 12. 실행 순서와 의존성

| 순서 | 단계 | 실험 ID | 시작 조건 | 종료 조건 |
|---:|---|---|---|---|
| 1 | 데이터 계약 | `000` | v2 schema 확정 | 불변식·누수검사 통과 |
| 2 | 기준선 | `001` | `D1-pilot` 생성 | 규칙/EWMA/CUSUM 결과 고정 |
| 3 | 핵심 진단 | `010`, `011`, `013` | fault oracle 확인 | ID fault gate와 error 분석 완료 |
| 4 | 예측·soft sensor | `012`, `020`, `021` | feature-lineage audit | threshold 인접 성능·coverage 통과 |
| 5 | 예지보전 | `030`, `031` | D3 scalar-fouling·EOL·censor 감사 통과 | health/RUL gate 통과; campaign-RUL은 별도 계약 |
| 6 | 일반화 | `040`, `041`, `050` | ID 모델 freeze | OOD·calibration·fallback 규칙 고정 |
| 7 | 제어 기준선 | `060` | 독립 `D4-control` | 안전 hard constraint 통과 |
| 8 | ML 제어 | `061` | forecast+uncertainty+fallback 통과 | MPC gate 통과 |
| 9 | 선택 연구 | `062` | constrained MPC보다 추가 가설 존재 | stress rollout gate 통과 |
| 10 | 통합 | `070` | 앞 단계 모델·threshold freeze | shadow-mode stress suite 통과 |

중간 실험이 합격하지 못하면 후속 모델을 억지로 진행하지 않는다. 특히 `000` 실패 시 모든 ML 실험, `040/050` 실패 시 ML 제어 실험을 중단한다.

## 13. 전체 합격 기준

개별 카드의 수치와 별도로 다음 조건을 모두 만족해야 해당 ML 계열을 `시뮬레이션 검증 완료`로 표시한다.

1. 데이터·split·라벨 manifest와 체크섬이 재현된다.
2. test window, profile, seed, counterfactual pair 누수가 0건이다.
3. 후속 일반 실험은 5 seeds, FlowTwin v0.2는 등록 3 seeds 전부와 profile-cluster 95% CI를 포함한다.
4. 안전관련 지표는 평균뿐 아니라 class별·profile별 worst case를 공개한다.
5. ML safety 판정은 deterministic rule보다 false-safe를 늘리지 않는다.
6. OOD 또는 낮은 confidence에서 보수적 fallback이 작동한다.
7. 제어 실험에서 ML은 FDV와 안전 인터록을 직접 override하지 않는다.
8. 합격하지 못한 시나리오와 실패한 seed도 삭제하지 않고 결과표에 남긴다.
9. 모델 크기, p50/p95/p99 추론시간, 메모리, 입력 센서 의존성을 보고한다.
10. 문서 상태는 `시뮬레이션 검증`을 넘지 않으며 실제 공장 성능으로 표현하지 않는다.

## 14. 실험 기록 템플릿

실행 시 아래 카드를 복사하고 기존 수치를 덮어쓰지 말고 새 run ID를 추가한다.

```text
실험 ID:
run ID / 작성일:
상태: 예정 | 실행 중 | 완료 | 실패
가설:
시뮬레이터 버전 / source checksum:
데이터셋 manifest / split manifest:
입력 신호군 / forbidden feature audit:
label 정의 / horizon / window / stride:
모델 / 하이퍼파라미터 / 학습 seed:
validation에서 정한 threshold:
ID 결과 + 95% CI:
OOD-profile / fault / combination / sensor 결과:
baseline과 paired 차이:
ablation:
누수·calibration·failure-case 점검:
합격 기준 판정:
산출물 경로:
해석과 다음 실험:
```

## 15. 바로 실행할 첫 묶음

2026-07-27 기준 우선 실행 묶음의 현재 상태는 다음과 같다.

1. `EXP-HTST-ML-000` / `D1-pilot`: **완료**. 12 profiles, 60 counterfactual groups, 1,200 episodes, signal/label 각 4,320,000 rows를 생성했고 train 7 / validation 2 / test 3 profile split과 누수·계약 감사 30/30을 통과했다.
2. `EXP-HTST-ML-001`: **D1 전체 기준선 실행 완료**. Rule/EWMA/CUSUM이 4,320,000 score rows를 처리했고 validation-only threshold·test-ID event/false-alarm/unsafe-volume artifact를 고정했다.
3. `D2-ood-dev`: **생성·split·감사·cache v0.3 완료**. ID 12/OOD 4 profiles, 48 groups, 960 episodes, 1,728,000 rows; `OOD_LOW_FLOW`/`OOD_HIGH_FLOW_WARM_FEED`; profile hash·33-parameter range·strict support-gap 포함 30/30 pass.
4. FlowTwin benchmark v0.2: **코드·계약 구현 완료, 16-variant×3-seed full matrix 결과는 이 문서의 근거에서 제외**. Cache v0.3, mask/loss ownership, event-aware sampler, six baselines, nine ablations, block conformal, profile metrics/bootstrap, tier firewall을 구현했다.
5. FlowTwin v0.3 후보: **구현·W96/1-seed opened-D2 post-hoc 평가·TCN validation-only alarm 감사 완료**. 네 모델 diagnostic을 고정했고, operational gate는 Hybrid/FlowTwin/DSPR만 통과했으며 TCN은 `0/75`로 실패했다. v0.2 registry에는 추가하지 않았다.
6. `D3-RUL`: **축소 smoke 완료, 정규 결과 미고정**. EOL persistence, right-censor, profile/OOD split, train-only 건강도·RUL 기준선 확인

현재 근거는 데이터·split·감사·기준선과 benchmark 소프트웨어 계약까지다. D1은 ID-only/test-ID 3 profiles이고, D2는 synthetic support-shift/test-ID 3·test-OOD 4 profiles의 이미 열린 개발 set이며 전체 neural matrix도 끝나지 않았다. 따라서 v0.3 후보를 포함해 novelty, 우월성, external-OOD, external-confirmatory 또는 현장 성능을 주장하지 않는다.

## 16. 구현된 실행 경로

| 파일 | 역할 | 현재 상태 |
|---|---|---|
| `ml_contract.json` | canonical taxonomy, alias, 구현 여부, feature allowlist, forbidden feature 계약 | 구현 |
| `generate_ml_dataset.py`, `ood_profile_contract.json` | profile·seed·onset·severity 무작위화, mode-matched pairs, ID + two strict-gap synthetic OOD domains, signal/oracle 분리, manifest/checksum | 구현·D1/D2 생성 |
| `split_ml_dataset.py` | `plant_profile_id` 단위 결정적 60/20/20 ID split, 모든 non-ID profile의 `test_ood_profile` 격리 | 구현·D1/D2 split |
| `audit_ml_dataset.py` | checksum, schema, forbidden feature, row-key, profile/config/seed/counterfactual 누수, D2 domain/hash/range/support-gap 감사 | 구현·D1/D2 30/30 pass |
| `run_ml_baselines.py` | 허용 신호만 사용하는 rule·EWMA·CUSUM, validation threshold 고정, event recall·지연·오경보·경보 전 unsafe volume | 구현 |
| `tests/test_ml_pipeline.py` | 생성 결정성, 20-class/alias 등가성, M01/F12 mode-matched timing, 누수 주입 검출, 기준선 산출물 회귀시험 | 구현 |
| `d3_rul_contract.json`, `d3_rul_common.py` | 독립 생산열화 범위, H0/H1 feature, EOL·censor·split 계약 | 구현 |
| `generate_d3_rul_dataset.py` | scalar-fouling 장기 trajectory, matched clean reference, health/EOL/censor label과 evidence | 구현 |
| `split_d3_rul_dataset.py`, `audit_d3_rul_dataset.py` | profile 단위 60/20/20+OOD split과 trace 기반 EOL/censor 재구성 감사 | 구현 |
| `run_d3_rul_baselines.py` | train-only constant/ridge health와 naive/KM/ridge-slope RUL 기준선 | 구현 |
| `tests/test_d3_rul_pipeline.py` | 결정성, event/censor, 누수·split·evidence 변조와 no-event train 실패 회귀시험 | 구현 |
| `lifecycle.py`, `maintenance_policy.json` | 반복 CIP·정비·교체, competing event와 service-interval censoring | 시뮬레이션 구현; ML benchmark 미실행 |
| `generate_multicycle_rul_dataset.py` | v3 multi-cycle trace/event/interval/schema/manifest/checksum 생성 | 구현 |
| `run_digital_twin.py`, `digital_twin_config.json` | 공정·교정·PLC·HACCP·lifecycle 통합 artifact 생성 | 구현 |
| `tests/test_digital_twin.py` | v3 결정성, checksum, 증거사슬, 바이오 제외와 원자적 실패 계약 | 구현 |
| `flowtwin_guard/graph.py` | 22-node/23-edge P&ID allowlist graph, profile별 exact FIFO overlay, observable route gate | 구현 |
| `flowtwin_guard/data.py` | train-only scaling, `valid_mask`/owning `eval_mask`, event landmark를 loss region에 보존하는 sampler, N00/M02 counterfactual pairing | 구현 |
| `flowtwin_guard/cache.py`, `build_flowtwin_cache.py` | v0.3 episode materialization, dataset/split/checksum/scaler와 `ml_pipeline_common.py`를 포함한 source provenance fail-closed 검증 | 구현·D1/D2 cache v0.3 생성 |
| `flowtwin_guard/model.py` | causal nominal Observer, uncertainty residual, advective-delay/hydraulic-control dual relation, 계층 head | 구현 |
| `flowtwin_guard/hybrid.py` | causal TCN + persistence-skip Observer residual + route-gated `V/Q` graph, sigmoid gate, taxonomy fault-mask hierarchy, physical OOD score | v0.3 개발 후보 구현 |
| `flowtwin_guard/alarm.py` | validation-only hysteresis·assert/clear persistence·cooldown grid, profile별 false-alarm/recall 제약, fail-closed selection | 구현 |
| `flowtwin_guard/baselines.py` | TCN, causal Transformer, static P&ID-GNN, learned dynamic GNN, Twin-residual TCN | 구현 |
| `flowtwin_guard/dspr.py` | 가장 가까운 선행연구 DSPR의 six-output causal diagnostic adaptation; exact reproduction 금지 metadata | 구현 |
| `flowtwin_guard/ablation.py` | `no_delay`부터 `no_status_relays`까지 등록 single-axis 9 ablations | 구현 |
| `flowtwin_guard/conformal.py` | validation counterfactual-group/observable-mode block-max conformal prediction set와 OOD abstention | 구현 |
| `flowtwin_guard/metrics.py` | test-ID macro-F1, one-to-one event, horizon-penalized latency, false alarm, selective/OOD 지표 | 구현 |
| `flowtwin_benchmark_contract.json`, `run_flowtwin_benchmark.py` | v0.2 16 variants×3 seeds frozen registry 보존, train/validation/test firewall, profile-level paired bootstrap, opt-in v0.3 candidate·동일 alarm policy·point-threshold/operational endpoint 분리 | 구현; full matrix 결과 미기재 |
| `flowtwin_v03_candidate_contract.json` | opened-D2 checksum, exact W96 budget, train-only delay gate, validation alarm과 post-hoc claim boundary 결박 | 구현·development-only |
| `audit_flowtwin_v03_alarm.py` | 동결 TCN checkpoint의 validation-only operational-grid 재감사, test iterator 미생성·`no_operating_point` checksum 근거 | 구현·TCN `0/75` 고정 |
| `report_flowtwin_v03_development.py` | 분리된 4-model diagnostic, validation feasibility, 조건부 3-model operational 결과의 결박·보고 | 구현; `FlowTwin-v03-D2-development-report` 재현 경로 |
| `train_flowtwin_guard.py` | 개별 FlowTwin 파이프라인 CLI; v0.2 정식 비교는 benchmark runner를 사용 | 구현·pre-audit run 보존 |
| `tests/test_flowtwin_guard.py`, `tests/test_flowtwin_benchmark.py`, `tests/test_flowtwin_metrics.py`, `tests/test_flowtwin_dspr.py`, `tests/test_flowtwin_hybrid.py`, `tests/test_flowtwin_alarm.py` | fractional delay·route·mask/loss·causality·DSPR·hybrid hierarchy·경보 firewall·conformal·metric·runner artifact 회귀시험 | 구현 |

정규 실행 명령은 다음 순서를 고정한다. 각 출력 디렉터리는 provenance 보존을 위해 비어 있어야 한다.

```bash
python3 generate_ml_dataset.py \
  --output ml_datasets/D1-pilot \
  --dataset-version D1-pilot \
  --profiles 12 \
  --ood-profiles 0 \
  --replicates 5 \
  --duration-s 1800 \
  --dt-s 0.5 \
  --seed 20260725

python3 split_ml_dataset.py \
  --dataset ml_datasets/D1-pilot \
  --output ml_datasets/D1-pilot-splits \
  --seed 20260725 \
  --train-ratio 0.60 \
  --validation-ratio 0.20

python3 audit_ml_dataset.py \
  --dataset ml_datasets/D1-pilot \
  --splits ml_datasets/D1-pilot-splits \
  --report ml_results/D1-pilot-audit.json

python3 run_ml_baselines.py \
  --dataset ml_datasets/D1-pilot \
  --splits ml_datasets/D1-pilot-splits \
  --output ml_results/D1-pilot-rule-baselines \
  --ewma-alpha 0.20 \
  --cusum-drift 0.50 \
  --quantile 0.995 \
  --reservoir-size 200000 \
  --seed 20260725

python3.12 -m venv .venv
.venv/bin/pip install -r requirements-ml.txt

.venv/bin/python build_flowtwin_cache.py \
  --dataset ml_datasets/D1-pilot \
  --splits ml_datasets/D1-pilot-splits \
  --output ml_datasets/D1-pilot-cache-v0.3 \
  --feature-set S3-context

.venv/bin/python run_flowtwin_benchmark.py \
  --dataset ml_datasets/D1-pilot \
  --splits ml_datasets/D1-pilot-splits \
  --cache ml_datasets/D1-pilot-cache-v0.3 \
  --contract flowtwin_benchmark_contract.json \
  --output ml_results/FlowTwin-Benchmark-D1-v0.2-full \
  --variants all \
  --seeds 20260727 20260728 20260729 \
  --window-size 64 \
  --stride 32 \
  --train-windows-per-episode 12 \
  --batch-size 32 \
  --observer-epochs 3 \
  --epochs 10 \
  --hidden-dim 32 \
  --observer-hidden-dim 48 \
  --layers 2 \
  --attention-heads 4 \
  --dropout 0.1 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --alpha 0.1 \
  --ood-alpha 0.01 \
  --device cpu \
  --save-predictions
```

위 D1은 ID-only이므로 전체 16개 variant×3 seeds와 frozen budget을 지켜도 tier는 `pilot`이다. `protocol_complete_synthetic`이 되려면 동일 계약에 검증된 `test_ood_profile` domain이 필요하며, 그 label도 external-confirmatory 증거는 아니다.

코어 구현 경로만 빠르게 재검산할 reduced development 명령은 다음과 같다. 이는 현재 v0.3 cache provenance에서 재실행하는 명령이며, 정식 성능 결론에 쓰지 않는다.

```bash
.venv/bin/python run_flowtwin_benchmark.py \
  --dataset ml_datasets/D1-pilot \
  --splits ml_datasets/D1-pilot-splits \
  --cache ml_datasets/D1-pilot-cache-v0.3 \
  --contract flowtwin_benchmark_contract.json \
  --output ml_results/FlowTwin-Benchmark-D1-core-dev-v0.3 \
  --variants flowtwin_guard tcn dspr_diagnostic_adaptation \
  --seeds 20260727 \
  --window-size 32 \
  --stride 32 \
  --train-windows-per-episode 4 \
  --batch-size 64 \
  --observer-epochs 1 \
  --epochs 1 \
  --hidden-dim 16 \
  --observer-hidden-dim 24 \
  --layers 1 \
  --attention-heads 4 \
  --dropout 0.1 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --alpha 0.1 \
  --ood-alpha 0.01 \
  --device cpu
```

생성·split·감사가 완료된 D2 synthetic OOD set을 v0.3 cache로 materialize한 후, 등록 행렬을 돌리는 정확한 명령은 다음과 같다. 데이터 생성 세 명령의 산출물은 이미 존재하므로 재현을 의도하지 않는다면 다시 덮어쓰지 않는다.

```bash
python3 generate_ml_dataset.py \
  --output ml_datasets/D2-ood-dev \
  --dataset-version D2-ood-dev \
  --profiles 12 \
  --ood-profiles 4 \
  --ood-contract ood_profile_contract.json \
  --replicates 3 \
  --duration-s 900 \
  --dt-s 0.5 \
  --seed 20260731

python3 split_ml_dataset.py \
  --dataset ml_datasets/D2-ood-dev \
  --output ml_datasets/D2-ood-dev-splits \
  --seed 20260731 \
  --train-ratio 0.60 \
  --validation-ratio 0.20

python3 audit_ml_dataset.py \
  --dataset ml_datasets/D2-ood-dev \
  --splits ml_datasets/D2-ood-dev-splits \
  --report ml_results/D2-ood-dev-audit.json

.venv/bin/python build_flowtwin_cache.py \
  --dataset ml_datasets/D2-ood-dev \
  --splits ml_datasets/D2-ood-dev-splits \
  --output ml_datasets/D2-ood-dev-cache-v0.3 \
  --feature-set S3-context

.venv/bin/python run_flowtwin_benchmark.py \
  --dataset ml_datasets/D2-ood-dev \
  --splits ml_datasets/D2-ood-dev-splits \
  --cache ml_datasets/D2-ood-dev-cache-v0.3 \
  --contract flowtwin_benchmark_contract.json \
  --output ml_results/FlowTwin-Benchmark-D2-ood-dev-v0.2-full \
  --variants all \
  --seeds 20260727 20260728 20260729 \
  --window-size 64 \
  --stride 32 \
  --train-windows-per-episode 12 \
  --batch-size 32 \
  --observer-epochs 3 \
  --epochs 10 \
  --hidden-dim 32 \
  --observer-hidden-dim 48 \
  --layers 2 \
  --attention-heads 4 \
  --dropout 0.1 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --alpha 0.1 \
  --ood-alpha 0.01 \
  --device cpu \
  --save-predictions
```

Domain verifier는 `profiles.csv`의 실제 33개 physical parameter로 profile config hash를 재구성하고, 모든 값이 versioned domain range에 속하며 domain 사이에 strict closed-interval support gap이 있는지 확인한다. D2 cache v0.3은 960 episodes/1,728,000 rows와 train-only 756,000 scaling rows를 checksummed source provenance와 함께 materialize했다. 위 full command가 완주해 runner가 `protocol_complete_synthetic`을 부여하더라도 synthetic protocol completion일 뿐 external-confirmatory 성능이 아니다. 이 명령은 frozen v0.2 전체 행렬의 재현 절차이며 아래 v0.3 후보 근거와 분리한다.

v0.3 결과는 이미 test-ID/OOD를 열어 본 D2에서 W96·seed `20260727` 하나로 실행한 post-hoc 개발 평가다. 후보 계약은 budget과 D2 checksum을 고정한다. 후보 shard의 실제 출력은 `ml_results/FlowTwin-v03-D2-candidate-matched-dev`다.

```bash
.venv/bin/python run_flowtwin_benchmark.py \
  --dataset ml_datasets/D2-ood-dev \
  --splits ml_datasets/D2-ood-dev-splits \
  --cache ml_datasets/D2-ood-dev-cache-v0.3 \
  --contract flowtwin_benchmark_contract.json \
  --candidate-contract flowtwin_v03_candidate_contract.json \
  --output ml_results/FlowTwin-v03-D2-candidate-matched-dev \
  --variants flowtwin_hybrid_v03_dev \
  --device cpu \
  --save-predictions
```

단일 4-model atomic run은 TCN이 validation operational grid에서 통과점을 찾지 못해 publish되지 않는 것이 정상이다. 성공한 FlowTwin/DSPR은 같은 candidate contract의 단일-variant shard로 각각 `ml_results/FlowTwin-v03-D2-flowtwin-operational-dev`, `ml_results/FlowTwin-v03-D2-dspr-operational-dev`에 보존했다. 아래 `VARIANT`/`OUTPUT`에 두 쌍을 각각 대입한다.

```bash
.venv/bin/python run_flowtwin_benchmark.py \
  --dataset ml_datasets/D2-ood-dev \
  --splits ml_datasets/D2-ood-dev-splits \
  --cache ml_datasets/D2-ood-dev-cache-v0.3 \
  --contract flowtwin_benchmark_contract.json \
  --candidate-contract flowtwin_v03_candidate_contract.json \
  --output OUTPUT \
  --variants VARIANT \
  --device cpu \
  --save-predictions
```

TCN은 같은 W96 budget의 row/class diagnostic을 `ml_results/FlowTwin-v03-D2-tcn-raw-diagnostic-dev`에 먼저 고정한 뒤, 동결 checkpoint를 아래 validation-only 감사에 넘겼다. 감사 결과 `ml_results/FlowTwin-v03-D2-tcn-alarm-feasibility-dev`는 feasible `0/75`, `test_iterator_constructed=false`, `test_rows_seen=0`을 기록한다.

```bash
.venv/bin/python run_flowtwin_benchmark.py \
  --dataset ml_datasets/D2-ood-dev \
  --splits ml_datasets/D2-ood-dev-splits \
  --cache ml_datasets/D2-ood-dev-cache-v0.3 \
  --contract flowtwin_benchmark_contract.json \
  --output ml_results/FlowTwin-v03-D2-tcn-raw-diagnostic-dev \
  --variants tcn --seeds 20260727 \
  --window-size 96 --stride 32 --train-windows-per-episode 12 \
  --batch-size 32 --observer-epochs 3 --epochs 10 \
  --hidden-dim 32 --observer-hidden-dim 48 --layers 2 \
  --attention-heads 4 --dropout 0.1 --learning-rate 0.001 \
  --weight-decay 0.0001 --alpha 0.1 --ood-alpha 0.01 \
  --device cpu --save-predictions

.venv/bin/python audit_flowtwin_v03_alarm.py \
  --raw-result ml_results/FlowTwin-v03-D2-tcn-raw-diagnostic-dev \
  --dataset ml_datasets/D2-ood-dev \
  --splits ml_datasets/D2-ood-dev-splits \
  --cache ml_datasets/D2-ood-dev-cache-v0.3 \
  --output ml_results/FlowTwin-v03-D2-tcn-alarm-feasibility-dev \
  --benchmark-contract flowtwin_benchmark_contract.json \
  --candidate-contract flowtwin_v03_candidate_contract.json
```

다음 diagnostic 표의 event·FA·unsafe는 validation-fitted row threshold를 쓴 **state-machine 적용 전** combined test 결과다. `전체` macro-F1은 ID/OOD row를 합친 descriptive pooled 값이지 추론 단위가 아니다.

| 모델 | parameter | 전체 / ID / OOD macro-F1 | validation-fitted row event-F1 / recall | row-threshold FA h⁻¹ | row-threshold unsafe L | OOD AUROC / FPR95 |
|---|---:|---:|---:|---:|---:|---:|
| FlowTwin-Hybrid v0.3 | 58,315 | 0.56218 / **0.68129** / 0.53188 | 0.03452 / 0.92529 | 186.51 | 414.37 | 0.55268 / 0.91749 |
| FlowTwin-Guard | 42,508 | 0.42491 / 0.55566 / 0.37698 | 0.02686 / 0.92816 | 245.83 | 0.00 | **0.76773** / 0.93618 |
| TCN | 15,379 | 0.58313 / 0.64073 / 0.54890 | 0.02222 / 0.91667 | 285.31 | 56.31 | 0.45451 / 0.98143 |
| DSPR adaptation | 54,166 | **0.59399** / 0.61125 / **0.58642** | **0.31215** / 0.80460 | **5.24** | 4,081.49 | 0.57475 / 0.93870 |

| 모델 | validation 유효/전체 | 선택 on/off/assert | validation event-F1 / recall | 최소 profile recall | 최대 profile FA h⁻¹ |
|---|---:|---:|---:|---:|---:|
| FlowTwin-Hybrid v0.3 | 4/75 | 0.70/0.20/5 s | 0.88889/0.80808 | 0.79167 | 3.40089 |
| FlowTwin-Guard | 4/75 | 0.50/0.10/5 s | 0.89583/0.86869 | 0.79167 | 4.29159 |
| TCN | **0/75** | — | — | — | — |
| DSPR adaptation | 8/75 | 0.70/0.05/2 s | 0.89362/0.84848 | 0.79167 | 3.88673 |

Operational test와의 교차는 **validation gate를 통과한 세 모델에만 조건부**로 보고한다.

| 모델 | 전체 / ID / OOD event-F1 | 전체 recall | 전체 / ID / OOD FA h⁻¹ | 전체 / ID / OOD unsafe L |
|---|---:|---:|---:|---:|
| FlowTwin-Hybrid v0.3 | 0.87690 / **0.90775** / 0.85841 | 0.91092 | 5.744 / 3.164 / 7.722 | 436.18 / 436.18 / 0.00 |
| FlowTwin-Guard | **0.90040** / 0.88462 / **0.91156** | **0.97414** | 7.770 / 6.710 / 8.582 | **10.51** / 10.51 / 0.00 |
| DSPR adaptation | 0.88438 / 0.86923 / 0.89474 | 0.81322 | **3.276** / **2.883** / **3.577** | 3,674.27 / 2,118.45 / 1,555.83 |

Hybrid은 ID macro-F1만 1위이고 전체/OOD macro-F1, row-threshold event-F1, operational 전체/OOD event-F1 전반에서 우월하지 않다. Validation 최대 profile FA `3.40/4.29 h⁻¹`를 통과한 Hybrid/FlowTwin의 test-profile 최댓값은 `9.42/9.09 h⁻¹`로 증가해 FA 제약이 일반화되지 않았다. DSPR은 test-profile 최댓값 `4.18 h⁻¹`과 가장 낮은 pooled FA를 유지했지만 recall `0.81322`, unsafe `3,674.27 L`를 동반했다. `F05` recall은 전 모델 `0`, Hybrid `F09` recall도 `0`이며, 네 모델의 conformal `DIAGNOSE`는 모두 `0`이다. 병렬 CPU 경합 때문에 wall time/throughput은 모델 효율 비교에 쓰지 않고 parameter count만 보고한다.

표준 보고서는 다음처럼 재현한다.

```bash
.venv/bin/python report_flowtwin_v03_development.py \
  --candidate ml_results/FlowTwin-v03-D2-candidate-matched-dev \
  --flowtwin ml_results/FlowTwin-v03-D2-flowtwin-operational-dev \
  --tcn ml_results/FlowTwin-v03-D2-tcn-raw-diagnostic-dev \
  --dspr ml_results/FlowTwin-v03-D2-dspr-operational-dev \
  --tcn-alarm-audit ml_results/FlowTwin-v03-D2-tcn-alarm-feasibility-dev \
  --output ml_results/FlowTwin-v03-D2-development-report
```

이 산출물은 우월성·external-OOD·confirmatory 근거가 아니다.

`ml_results/FlowTwin-Benchmark-D2-core-budget-v0.2/`의 부분집합 실행은 frozen 비시드 budget을 그대로 쓰되 FlowTwin/TCN/DSPR와 seed `20260727`만 선택했다. 따라서 tier는 `development`다. 전체 macro-F1은 `0.44545/0.56266/0.58609`, OOD AUROC는 `0.81138/0.41887/0.58005`, event-F1은 `0.01872/0.03626/0.25665`, false-alarm onset은 `347.85/169.85/3.09 h⁻¹`였다. FlowTwin은 OOD AUROC가 가장 높았지만 FPR95 `0.95059`, class/event 열세와 가장 많은 오경보를 동반했다. DSPR은 class/event와 오경보에서 가장 좋았지만 경보 전 unsafe volume이 `6,004.27 L`로 FlowTwin `150.30 L`보다 컸다. 세 모델 모두 singleton `DIAGNOSE=0`이며 평균 conformal set은 10.34–12.60 classes다. 어느 모델도 현재 합격 또는 우월성 결과로 판정하지 않는다.

`signals.csv`에는 조인·평가용 context와 `S3-context` allowlist만 저장된다. `plant_mode`는 context이지 기본 feature가 아니며, 기준선은 대신 `cip_cycle_active`로 gating한다. 실제 `sensor_available`·`power_available`·밸브 고장계수·세정효율·FDV 내부 위치·balance-tank moment·CIP soil·shadow-HX state·`alarm_count`·`alarm_unsafe_forward`·실제 `differential_pressure_bar`가 들어오면 감사와 기준선 실행이 모두 실패해야 한다. 관측 가능하다고 계약한 온도·유량·압력·CIP conductivity/pH 센서, 상태 relay, `fdv_position_feedback`, command, position error만 입력으로 허용한다. `oracle_labels.csv`는 학습 입력 파일이 아니며, 평가 시 `(episode_id, time_start_s, time_s)`로만 결합한다.

2026-07-25에 재생성한 v2.2 D1 축소 산출물은 기존 경로명을 유지한 `ml_results/D1-v2-smoke/`에 있다. 4 profiles × 20 canonical class × 1 replicate × 120초(`dt=0.5초`)로 실행해 4개 counterfactual group, 80 episodes, `signals`/`oracle_labels` 각 19,200행을 생성했다. Profile split은 train 40 / validation 20 / test 20 episodes다. 누수·계약 감사 **30/30**은 실패 0건이었고 rule·EWMA·CUSUM 세 경로가 전 19,200행을 처리해 240 episode-method 평가행을 만들었다. 정상 4 episodes와 짧은 trace의 `F11` 한 건은 episode 종료 전에 평가 가능한 고장 관측효과가 없어 `detection_eligible=0`으로 보존하고 지연 평가에서 제외했다. 별도 1 profile × 20 class × 1,800초 / 0.5초 = 72,000행 probe에서는 모든 고장 class가 허용 신호의 관측효과 시점을 얻었다.

FlowTwin-Guard의 동일 D1 축소자료 실행은 `ml_results/FlowTwin-Guard-v0.1-balanced-smoke/`에 model/checkpoint, graph/scaler/calibration state, test prediction 4,800행, metrics, manifest와 checksum을 발행했다. 이 실행은 **코드 경로를 점검하는 legacy smoke**이며 2개 train profile, 짧은 관측구간과 20 diagnosis epochs의 결과다. Test row accuracy 0.657708, 20-class macro-F1 0.353002, anomaly F1 0.485394였다. Event recall 17/17과 detection 전 unsafe 0 L는 false-alarm steps/negative-hour 3430.697674를 동반해 합격 결과가 아니다. Nominal 90% conformal의 profile-disjoint test coverage는 0.850208, 평균 set 크기는 7.46이었고 singleton 진단은 없었다. 이 결과는 v0.2 baseline/ablation 비교에 사용하지 않는다.

`ml_results/FlowTwin-Guard-D1-seed20260727/`의 D1 test-ID row accuracy `0.26423`과 nominal-90% empirical row coverage `0.64940`도 **pre-audit direct run**이다. 이후 `valid_mask & eval_mask` loss ownership, event landmark owning-region 보장, counterfactual-group/observable-mode block-max conformal, cache materialization provenance, DSPR 비교, profile 집계·tier 판정을 교정했으므로 pipeline-only로 보존하고 v0.2 성능표에서 제외한다. `ml_results/FlowTwin-Benchmark-D1-core-dev-v0.2/`는 FlowTwin/TCN/DSPR 3 variants×1 seed의 reduced `development` 경로를 완주했지만, `ml_pipeline_common.py` source provenance가 누락된 구 cache v0.2를 사용했다. 따라서 역사적 pipeline 기록으로만 남기고 현재 v0.3 cache 명령으로 재실행해야 한다. 구조와 정확한 실행계약은 [FLOWTWIN_GUARD.md](FLOWTWIN_GUARD.md)를 따른다.

v2.2 D3 축소 산출물은 `d3_results/D3-RUL-v2.2-smoke/`에 있다. 6 profiles × 4 trajectories = 24 trajectories, ID 범위와 겹치지 않는 OOD profile 1개, signal/label 각 990행, EOL 13건, right-censor 11건을 포함한다. Split seed 456으로 train 12 / validation 4 / test-ID 4 / test-OOD 4 trajectories를 격리했고, 감사 **26/26**은 실패 0건이었다. 모든 기준선 파라미터는 train-ID 12 trajectories에서만 적합했으며 전체 split에 175 RUL landmark를 출력했다. 이 데이터는 acceleration 100, 최대 simulator duration 1,200초, 생성 seed 123인 재현 smoke이며, 실제 정비수명, 현장 일반화 또는 정규 D3 benchmark 결과로 해석하지 않는다. 전체 회귀시험의 최신 결과는 `python3 -m unittest discover -s tests -v`의 실행 기록을 기준으로 한다.
