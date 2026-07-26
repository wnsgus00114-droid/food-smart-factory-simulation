# 우유 HTST v2.2 + 비바이오 통합계층 v3 외부 근거와 모델 추적성

v3 추가 자료의 최종 접근일은 **2026-07-27**, 기존 v2.2 자료는 **2026-07-25**다. 이 문서는 공개 자료에서
확인한 사실과 시뮬레이터의 공학 가정을 구분하고, 각 자료가 어느 코드 요소에
영향을 주었는지 기록한다.

이 자료들은 연구용 일반 공정 surrogate와 공개 참조 P&ID·PLC·교정·HACCP·수명 계층을 구성하는 근거다. 특정 공장의 as-built P&ID, 설비 명판·성능곡선, 계측 교정자료, 추적자 시험 또는 관할기관의 승인을 대체하지 않는다. 바이오 모델은 사용자 결정에 따라 범위에서 제외했다. 따라서 v2.2/v3는 실제 공장 디지털 트윈이나 HACCP, 법규, 제품 안전·출하 또는 세정 유효성 판정 도구가 아니다.

## 자료 등급

- FDA 문서는 미국의 공식 NCIMS 모델 문서다. 현 시뮬레이터에는 규제 검토
  체크리스트로만 사용했고, 적합 판정을 내리지 않았다.
- Tetra Pak *Dairy Processing Handbook*은 설비 제조사가 공개한 공정 기술
  핸드북이다. 대표 공정과 설계 예시를 구성하는 데 사용했지만 법령이나 특정
  공장 실측자료로 취급하지 않았다.
- 학술 논문은 동적 모델 구조의 1차 문헌이다. 논문의 실험장치와 현재 가상
  설비가 같지 않으므로 논문 계수를 그대로 이식하거나 검증 결과를 전이하지
  않았다.

## 1. 공식 규제 참조

### FDA — *Grade “A” Pasteurized Milk Ordinance (Grade “A” PMO), 2025 Revision*

- 공식 목록: <https://www.fda.gov/food/milk-guidance-documents-regulatory-information/national-conference-interstate-milk-shipments-ncims-model-documents>
- 2025 PMO PDF: <https://www.fda.gov/media/193438/download>
- 확인 사항: FDA의 NCIMS 모델 문서 목록은 2025 PMO를 2026년 6월 공개본으로
  제공한다.
- v2 연결: HTST의 기록, holding, 안전장치 및 flow diversion에 관한 향후 규제
  추적성 점검의 공식 문서 기준으로 보존한다. 현재 `model.py`의 설정값, 인터록
  임계값, 알람 로직 또는 CIP 조건을 이 문서에 적합하다고 판정하지 않았다.
- 적용 한계: PMO는 미국 모델 문서이며 대한민국 식품공전이나 국내 HACCP
  기준을 대신하지 않는다. PMO를 인용한다는 사실만으로 공정·장비 승인이나
  제품 출하 적합성이 성립하지 않는다.

## 2. 공개 공정·설비 근거

### Tetra Pak — *Dairy Processing Handbook: Designing a Process Line*

- URL: <https://dairyprocessinghandbook.tetrapak.com/chapter/designing-process-line>
- 공개 자료에서 확인한 내용:
  - 일반 우유 살균 예시는 balance tank, feed pump, 재생 예열, 최종 가열,
    holding tube, booster pump, FDV, 재생 냉각과 최종 냉각으로 이어진다.
  - 자료의 대표 설계 예시는 처리량 20,000 L/h, 온도 프로그램 4–72–4°C,
    재생 효과 90–94%를 사용한다. 이는 예시값이지 보편 규정값이 아니다.
  - 살균된 제품 측 압력을 원유·유틸리티 측보다 높게 유지하기 위해 booster
    pump와 압력 계측을 사용하는 설계 개념을 설명한다.
  - 온도·holding time·압력차 조건이 충족되지 않을 때 FDV로 제품 탱크 유입을
    막고 balance tank로 회송하는 구성을 설명한다.
  - FDV 위치로 (1) holding tube 직후와 (2) 냉각부 이후의 두 대안을 제시한다.
    두 번째 대안은 온도 이탈 후 설비 비움·세정·살균 절차를 수반한다.
  - 전진 제품은 holding 이후 재생부에서 열을 회수한 뒤 최종 냉각되고, 제품
    출구 온도는 냉각 유틸리티 제어와 함께 기록되는 구성을 설명한다.
- v2 연결:
  - 주 시나리오 모델은 **holding tube 직후 FDV**인 첫 번째 대안을 명시적으로
    선택한다. 전진 branch만 post-FDV 고정재고 FIFO를 거쳐 재생 hot side와
    cooler를 통과하고, divert branch는 balance-tank 회송 경계로 보낸다.
  - 측정 온도, 추정 체류시간, 측정 압력차, 누설 신호와 밸브 피드백을 전진
    permissive 및 알람에 사용한다. 실제 안전상태는 별도 oracle로 유지한다.
  - 기본 `nominal_flow_l_h=20_000`, `raw_milk_temp_c=4`,
    `regenerator_effectiveness=0.90`, `final_product_target_c=4`는 위 대표 예시를
    연구 기준점으로 선택한 것이다.
- 공학 가정: post-FDV 재고 5초, 압력손실 식, booster gain, 요구 압력차,
  센서 잡음·지연, FDV 이동시간과 누설률은 이 핸드북에서 가져온 설계값이
  아니다. 실제 배관·밸브·계측 자료가 없으므로 코드의 가상 설비 값이다.

### Tetra Pak — *Dairy Processing Handbook: Heat Exchangers*

- URL: <https://dairyprocessinghandbook.tetrapak.com/chapter/heat-exchangers>
- 공개 자료에서 확인한 내용:
  - 우유 HTST의 대표 범위를 72–75°C, 15–20초로 제시한다.
  - 재생 열교환은 효율적인 현대 설비에서 최대 약 94–95%까지 가능하다고
    설명한다.
  - holding time은 유량에 반비례하고, 비균일 속도분포 때문에 가장 빠른 유체도
    요구시간을 만족하도록 효율계수를 적용한다. 난류 holding tube의 계수 예시
    범위는 0.8–0.9이며 계산 예시는 0.85를 사용한다.
  - 유량, 온도차와 fouling은 열전달 성능과 운전 가능시간에 영향을 준다.
- v2 연결:
  - 기본 setpoint 74°C, 명목 holding 18초, 최소 holding 15초를 공개된 대표
    범위 안의 연구점으로 뒀다. 72°C diversion threshold는 모델 기준값이며
    특정 제품·관할 법규에 대한 판정값이 아니다.
  - `fastest_flow_efficiency=0.85`를 적용해 평균 parcel 체류시간과 가장 빠른
    흐름의 보수적 체류시간 proxy를 구분한다. 유량 변화와 정지 구간은 고정용적
    parcel queue에서 계산한다.
  - 재생부·가열부·냉각부와 fouling 영향은 주 모델에서 효과도와 동적 지연으로
    축약한다. `process_components.py`의 `DynamicHeatExchanger`는 양측 유체와
    벽체의 저장에너지, UA 저하와 외부 열손실을 계산하며 주 루프에서
    regenerator·heater·cooler 보존성 shadow network로 함께 실행된다.
- 공학 가정: 주 시나리오의 1차 지연 상수, stationary cooling 시정수,
  효과도 변화식 및 post-FDV 재고는 실험으로 식별하지 않았다. 0.85는 공개
  범위에서 택한 설계 proxy이며 실제 설비의 RTD 또는 추적자 시험을 대신하지
  않는다.

### Tetra Pak — *Dairy Processing Handbook: Tanks*

- URL: <https://dairyprocessinghandbook.tetrapak.com/chapter/tanks>
- 공개 자료에서 확인한 내용: balance tank는 펌프 흡입측 수두와 유량을
  안정시키며, 불충분하게 열처리되어 회송된 제품이 들어오면 원유 유입을 줄이고
  제품을 재순환시키는 구성을 설명한다. 세정액 순환에도 유사한 경로를 쓴다.
- v2 연결: `process_components.py`의 `BalanceTank`는 원유와 회송액을
  완전혼합하고, 온도·재통과 횟수·위험 부피·잔류 화학물질의 부피가중 모멘트와
  탱크 부피수지를 추적한다. overflow, 재고 부족 및 최소 운전수위 위반은
  fail-fast로 처리한다.
- 현재 통합 상태: 주 `model.py`는 전 timestep의 divert branch를 명시적
  return-line 재고로 보존한 뒤 다음 timestep에 `BalanceTank`로 회송한다.
  탱크의 혼합 온도·평균 재통과 횟수·위험/화학물질 분율이 다음 공정 유입에
  되먹임되며, fresh-feed·탱크·return-line을 포함한 외부 부피수지도 별도로
  기록한다.
- 공학 가정: tank 용량, 초기수위, 완전혼합 및 주변 열교환 시정수는 특정 설비
  자료로 식별한 값이 아니다.

### Tetra Pak — *Dairy Processing Handbook: Cleaning of Dairy Equipment*

- URL: <https://dairyprocessinghandbook.tetrapak.com/chapter/cleaning-dairy-equipment>
- 공개 자료에서 확인한 내용:
  - 유가공 CIP는 제품 회수·예비수세·세제 세정·수세 등의 순서로 운전하며,
    가열면은 단백질과 무기질 침착을 고려한 알칼리·산 세정이 필요하다.
  - 세정 결과에 영향을 주는 핵심 조건으로 세제 농도, 온도, 유속에 의한
    기계적 효과와 시간을 설명한다.
  - 유량, 온도, 시간과 전기전도도를 감시·기록하더라도 세정 효과의 별도 검증이
    필요하다고 명시한다.
- v2 연결:
  - 주 모델의 `cip_cycle`은 예비수세–가성–중간수세–산–최종수세 순서와
    온도·유량·농도 proxy, 재순환량 및 scalar fouling 감소를 기록한다.
  - `CIPSoilModel`은 주 `model.py`의 `cip_cycle`과
    `incomplete_cleaning` 실행 경로에 결합되어 단백질계와 무기질계 soil을
    분리하고 온도, 유속, 알칼리·산 조건, 수세 희석 및 잔류 화학물질 proxy를
    계산한다.
- 공학 가정: 코드의 phase 지속시간, 농도, 온도, 제거율과 clean 판정 임계값은
  핸드북의 예시 레시피를 검증된 SOP로 복사한 것이 아니다. 세제 공급사 지침,
  실제 회로 유속·전도도, 잔류물·ATP·미생물 검사를 대체할 수 없다.

## 3. 동적 모델과 fouling/CIP 1차 문헌

### Gutierrez, Diniz & Gut (2014)

- 논문: C. G. C. C. Gutierrez, G. N. Diniz, J. A. W. Gut,
  “Dynamic simulation of a plate pasteurizer unit: Mathematical modeling and
  experimental validation,” *Journal of Food Engineering*, 131, 124–134.
- DOI: <https://doi.org/10.1016/j.jfoodeng.2014.01.029>
- 1차 문헌의 범위: 보존·수송 방정식을 이용해 가열, 냉각, 재생의 세 판형
  열교환부와 비등온 holding tube를 포함한 동적 모델을 만들고, 실험실 규모
  장치의 기동과 유량 변화 응답으로 검증했다.
- v2 연결: transient 열상태, holding 수송, start/stop 및 에너지·부피 보존을
  명시적인 상태로 다뤄야 한다는 구조적 근거다.
  `DynamicHeatExchanger`의 유체·벽체 저장에너지와 수치적 에너지수지도 이
  원칙을 따른다.
- 전이하지 않은 것: 논문의 장치 치수, 열전달계수, 경계조건, 유한차분 격자와
  검증 결과를 현 가상 설비에 복사하지 않았다. 따라서 해당 논문의 실험 검증이
  v2 검증을 의미하지 않는다.

### Sharma & Macchietto (2021)

- 논문: A. Sharma, S. Macchietto, “Fouling and cleaning of plate heat
  exchangers: Dairy application,” *Food and Bioproducts Processing*, 126,
  32–41.
- DOI: <https://doi.org/10.1016/j.fbp.2020.12.005>
- 1차 문헌의 범위: 판형 열교환기의 동적·분포형 열수력 모델에 fouling 성장과
  CIP 제거 모델을 결합하고, 가열–세정 운전주기를 연결하는 구조를 제시한다.
- v2 연결: 열전달 성능, 압력손실, fouling과 세정을 한 상태계에서 연결하는
  연구 방향의 근거다. 주 모델의 제어 plant는 계산비용을 줄인 효과도·scalar
  fouling surrogate를 사용하고, 함께 실행되는 shadow 열교환기와 CIP soil
  상태는 UA 저하·에너지 보존·soil 제거를 별도로 제공한다.
- 전이하지 않은 것: 논문의 2차원 공간분포, deposit moving boundary, 장치별
  물성·kinetic 계수와 실험 적합값은 구현하지도, v2 기본값으로 사용하지도
  않았다.

### Zhang et al. (2026)

- 논문: Y. Zhang, F. W. J. van den Berg, M. L. Andersen,
  B. R. Parjikolaei, S. Bakalis, “Cleaning in place of plate heat exchangers –
  Removal behavior of whey protein fouling,” *Chemical Engineering Science*,
  320, 122395.
- DOI: <https://doi.org/10.1016/j.ces.2025.122395>
- 1차 문헌의 범위: 판형 열교환기 whey-protein fouling 세정에서 온도, 유속,
  pH와 초기 오염량의 영향을 측정하고, 후기 잔류 fouling 제거가 느려지는
  현상을 포함한 세정 모델을 제시·검증한다.
- v2 연결: `CIPSoilModel`이 온도·유속·화학조건과 잔류 soil을 분리해 계산하고,
  `incomplete_cleaning` 시나리오에서 잔류 오염을 명시적으로 시험하도록 한
  구조적 근거다.
- 전이하지 않은 것: 논문의 특정 whey fouling, plate geometry, NaOH 조건과
  fitted kinetics를 우유 HTST 전체 회로의 보편값으로 사용하지 않았다.

### Alvarez et al. (2010) — 산업 campaign 계측

- 논문 DOI: <https://doi.org/10.3168/jds.2009-2760>
- 1차 문헌의 범위: 산업용 관형 UHT 설비에서 launch, production, 분산 CIP,
  최종 rinse와 다음 생산까지의 온도·유량·차압·전도도·탁도 변화를 계측했다.
- v2.2 연결: `run_campaign.py`와 `HTSTSimulator.run_campaign()`은 생산–CIP–재기동을
  하나의 시간축으로 실행한다. 물·제품·약품 분율을 holding/FDV/post-FDV FIFO와
  balance tank에 전달하고, 전도도·pH 및 제품-interface proxy가 허용되기 전
  재기동 유체를 제품 경계에서 배출한다.
- 전이하지 않은 것: 논문의 디저트 크림, 관형 UHT 설비와 현 모델의 우유 PHE
  surrogate는 동일하지 않다. 논문에 보고된 endpoint 수치나 탁도·전도도 값을
  보편 임계값으로 복사하지 않았다.

### Gillham et al. (1999), Dürr (2002), Bremer et al. (2006) — 역사적 v2.2 기록

- 세정 단계·침식 거동: <https://doi.org/10.1205/096030899532420>
- 경험적 잔류오염 모델: <https://doi.org/10.1205/096030802321154745>
- 과거 검토한 CIP/biofilm 문헌: <https://doi.org/10.1016/j.ijfoodmicro.2005.07.004>. 사용자 결정에 따라 v3 바이오 구현 근거에서는 제외한다.
- v2.2 연결: protein/mineral soil 제거와 후기 잔류를 별도 상태로 보존하고,
  chemical-clear, soil-clean 및 `surface_hygiene_risk_fraction`을 같은 판정으로
  합치지 않는다. 완전 recipe와 불완전 recipe의 재기동 위험을 별도 캠페인으로
  비교한다.
- 적용 한계: 현재 `CIPSoilModel`은 논문의 다층 팽윤·침식 또는 biofilm kinetics를
  재현한 모델이 아니라 두 soil의 축약 surrogate다. 따라서 hygiene 결과는
  합성 oracle이며 미생물 세정검증이 아니다.

### Georgiadis & Macchietto (2000), Berce et al. (2025), Noot et al. (2025)

- fouling–열수력 결합: <https://doi.org/10.1016/S0009-2509(99)00429-7>
- 확률적 열교환기 RUL 구조: <https://doi.org/10.1016/j.applthermaleng.2025.126954>
- right-censored 산업 열화자료: <https://doi.org/10.36001/ijphm.2025.v16i2.4260>
- v2.2 연결: D3 파이프라인은 fouling, 열부하·압력/유량 및 정상 전진 유지능력을
  EOL evidence로 분리하고, 예방 종료는 `event_observed=0`으로 보존한다.
  censored trajectory에는 exact RUL을 쓰지 않고 관측 가능한 lower bound만 둔다.
- 전이하지 않은 것: 논문별 장치·고장 메커니즘·posterior 또는 fitted 계수를
  현 설비의 EOL/RUL 값으로 이식하지 않았다. D3 결과는 scalar fouling surrogate의
  합성 수명 실험이다.

## 4. 비바이오 통합계층 v3 근거

### P&ID와 PLC 표현

- ISO 10628-1:2014 공식 페이지: <https://www.iso.org/cms/%20render/live/en/sites/isoorg/contents/data/standard/05/18/51840.html?browse=tc>
  - 흐름도의 분류·내용·표현 원칙을 참조해 `reference_pid.json`과 GitHub Mermaid view를 구성했다.
  - ISO 표준 심볼 적합성, 설계 승인 또는 as-built 상태를 주장하지 않는다.
- IEC 61131-3:2025 공식 페이지: <https://webstore.iec.ch/en/publication/68533>
  - Structured Text가 programmable controller용 textual language라는 점을 기준으로 `plc/htst_reference.st`를 작성했다.
  - vendor compiler, hardware I/O, task scheduling, watchdog, FAT/SAT와 safety integrity 검증은 전이하지 않았다.

### 계측 교정과 불확실성

- NIST Technical Note 1297: <https://www.nist.gov/pml/nist-technical-note-1297>
  - 표준불확실성 성분을 RSS로 결합하고 `U = k × u_c`로 확장불확실성을 표현하는 방법을 `sensor_calibration.py`에 사용했다.
- NIST metrological traceability: <https://www.nist.gov/metrology/metrological-traceability>
  - traceability가 문서화된 교정 사슬을 요구한다는 claim boundary를 `sensor_catalog.json`에 고정했다.
  - 합성 as-found/as-left point, 센서 계수와 교정주기는 실제 certificate 값이 아니다.

### HACCP 기록 구조

- 대한민국 `식품 및 축산물 안전관리인증기준`, 식품의약품안전처고시 제2026-25호, 제6조: <https://law.go.kr/admRulLsInfoP.do?admRulSeq=2100000276712>
- Codex General Principles of Food Hygiene/HACCP publications: <https://www.fao.org/fao-who-codexalimentarius/publications/en/>
- FDA/NACMCF HACCP Principles & Application Guidelines: <https://www.fda.gov/food/hazard-analysis-critical-control-point-haccp/haccp-principles-application-guidelines>

세 자료에서는 위해분석, 중요관리점, 한계기준, 모니터링, 개선조치, 검증, 기록의 7원칙 구조만 사용했다. `haccp_plan.json`의 수치 한계는 현장 검증값이 아니고, `haccp.py`는 적합성 평가·전자서명·자동출하를 수행하지 않는다.

### 반복 정비와 virtual age

- M. Kijima, “Some Results for Repairable Systems with General Repair,” *Journal of Applied Probability* 26(1), 89–102 (1989): <https://doi.org/10.1017/S0021900200041826>
  - 일반 수리 후 virtual-age 구조를 `lifecycle.py`의 불완전 정비 상태전이에 사용했다.
  - `q`, Weibull, damage와 fouling 계수는 논문에서 복사하지 않고 합성 정책으로 분리했다.
- Sharma & Macchietto (2021): <https://doi.org/10.1016/j.fbp.2020.12.005>
  - 생산 fouling과 cleaning을 같은 반복 운전계에 연결하는 구조적 근거로 재사용했다.

`maintenance_policy.json`에 과거 잘못 연결됐던 NFF(no-fault-found) 논문 DOI는 제거하고 Kijima 1989의 1차 문헌 DOI로 교정했다.

## 5. 출처–구현 상태 요약

| 외부 근거 요소 | v2 코드 연결 | 상태 |
|---|---|---|
| 일반 HTST 공정 순서와 holding 직후 FDV 대안 | `model.py`의 holding → FDV line → FDV → post-FDV FIFO → 재생/냉각 | 주 시나리오 통합 |
| 유량에 따른 체류시간과 fastest-flow 효율계수 | parcel별 평균·fastest 체류시간, `fastest_flow_efficiency` | 주 시나리오 통합; RTD 실측 미검증 |
| 살균측 양압, booster와 압력 계측 | 실제·측정 압력차, 누설 proxy, permissive와 알람 | 주 시나리오 통합; 축약 수력식 |
| FDV 위치·이동·피드백 | 연속 밸브 위치, 이동시간, mismatch, slow/stuck/leakage 고장 | 주 시나리오 통합; 밸브 실측 미식별 |
| 전진 제품의 재생 냉각·최종 냉각과 재고 지연 | post-FDV 고정재고 FIFO와 제품 온도 계측 | 주 시나리오 통합; 효과도 surrogate |
| balance tank 회송과 혼합 | `process_components.py::BalanceTank` | 주 루프 통합; 1-step return-line 지연 가정 |
| 양측 유체·벽체 열저장과 fouling에 따른 UA 저하 | `process_components.py::DynamicHeatExchanger` | 주 루프의 보존형 shadow network로 통합; 제어 plant는 축약식 유지 |
| 알칼리·산 CIP, 유속·온도·농도·시간 | `cip_cycle`, `CIPSoilModel`, `incomplete_cleaning` | 시뮬레이션 구현; 세정 유효성 미검증 |
| 생산–CIP–재기동 연속 campaign | `run_campaign.py`, versioned state export/import, global campaign clock | 상태·RNG·FIFO·tank·HX·soil carryover 구현; 실제 CIP 회로 미식별 |
| 제품·물·약품 interface와 재기동 배출 | `product_fraction`, `chemical_fraction`, product-boundary restart guard | 보존형 tracer 구현; 실제 탁도/전도도 보정 미검증 |
| right-censored health/EOL/RUL | 독립 D3 generator·split·audit·baseline | 합성 장기수명 pipeline 구현; 현장 RUL 성능 미검증 |
| FDA PMO 요구사항 | 문서 추적성과 향후 검토 체크리스트 | 코드 적합성·규제 승인 미판정 |
| ISO 10628 흐름도 구조 | `reference_pid.json`, `reference_plant.py`, Mermaid renderer | 공개 참조 topology 구현; ISO 적합/as-built 아님 |
| IEC 61131-3 Structured Text | `plc/htst_reference.st`, cause/effect marker 검사 | vendor-neutral shadow 구현; 배포 PLC 아님 |
| NIST TN 1297 불확실성 표현 | `sensor_calibration.py`, `sensor_catalog.json` | 합성 교정 구현; traceability 없음 |
| HACCP 7원칙·기록 구조 | `haccp_plan.json`, `haccp.py` | fail-closed 증거계층 구현; 적합성·자동출하 없음 |
| Kijima general-repair virtual age | `lifecycle.py`, `maintenance_policy.json` | 반복 정비/CIP 구현; 현장 계수 미적합 |

## 6. 공개 사실이 아닌 주요 v2/v3 공학 가정

다음 값과 식은 출처가 존재한다는 이유만으로 확정된 공정값이 아니다.

- `pasteurization_setpoint_c=74`, `nominal_holding_time_s=18`,
  `minimum_holding_time_s=15`, `diversion_threshold_c=72`는 공개된 대표 범위에서
  고른 연구 설정이다. 대상 제품의 법적 한계나 검증된 scheduled process가 아니다.
- `fastest_flow_efficiency=0.85`는 공개 예시 계수다. 실제 holding tube의
  최단체류시간 분포, laminar/turbulent 전이 또는 axial dispersion을 식별한
  결과가 아니다.
- 요구 압력차 0.50 bar, booster gain, 배관 압력손실, 펌프효율, 센서 정확도와
  잡음, 지연, 밸브 이동·누설 파라미터는 모두 가상 설비의 공학 가정이다.
- 재생·가열·냉각 효과도와 시정수, 정지 유체의 주변 냉각 시정수, post-FDV
  5초 재고는 실제 UA, plate/channel 수, 배관 체적에서 동정하지 않았다.
- 상대 열처리 진단의 `z=7°C`와 기준시간·온도는 수치 실험용 무차원 proxy다.
  특정 병원성 또는 부패 미생물의 D/z 값, 사멸률이나 제품 안전확률이 아니다.
- fouling 증가율, 열전달·압력손실 영향계수, CIP 제거율, protein/mineral soil
  초기량과 clean threshold는 실험 적합값이 아니다.
- `soil_scale=(650+350)/0.65`, 65:35 생산 침착 조성, 제품/약품 interface
  허용값과 재기동 one-through drain 정책은 상태 연속성을 시험하기 위한 공학
  가정이다. 실제 회수·배수 valve topology나 제품손실 최적값이 아니다.
- D3의 EOL threshold, persistence, 열화 가속계수와 administrative censor horizon은
  합성 수명실험 계약이며 설비 제작사의 허용한계나 예방정비 주기가 아니다.
- 고장 시작시각, 지속시간, severity, drift/dropout/slow-valve/leakage 크기는
  ML·인터록 실험을 위한 합성 주입 조건이다.
- v3 센서 lag·gain·offset·drift·hysteresis·noise·dropout, guard band와 교정주기는 합성 계측 계약이다.
- v3 lifecycle의 Weibull shape/scale, damage/fouling rate, complete/incomplete CIP 제거율과 정비 action factor는 합성 정책이다.

## 7. 검증 한계와 실제 적용에 필요한 자료

현재 공개자료만으로 제거할 수 없는 불확실성은 다음과 같다.

- 실제 HTST as-built P&ID, FDV 실제 위치, holding tube 길이·내경·단열, 모든 구간의
  배관 체적과 정상·최대 유량
- 판형 열교환기의 plate/channel 배열, 구간별 UA, 양측 유량·온도, fouling에
  따른 열전달·압력손실 변화
- 펌프·밸브 성능곡선, fail-safe 회로, 실제 PLC cause-and-effect, 인터록 latch/reset,
  recorder-controller 구성과 실제 scan time
- 센서 종류·범위·정확도·traceable 교정기록, 응답시간, dropout 방식, 시간동기 품질
- 물 또는 제품을 이용한 추적자 시험 RTD, 기동·정지·정전·회송 실험 데이터
- 원유 조성·점도·밀도·비열의 로트별 변화. 바이오 kinetics와 challenge-study는 현재 범위에서 제외한다.
- 실제 CIP 회로, 세제·농도·온도·유속·전도도·pH 기록, 잔류물 검사 및 세정 validation 결과
- 고장·정비·교정·교체 event, right-censor 원인과 asset별 운전부하 이력
- 대한민국 최신 식품공전 원문과 제품별 예외, 국내 HACCP CCP 한계·감시·
  이탈조치에 대한 조항별 추적성 표

위 자료로 파라미터 식별, 독립 검증과 관할 규정 비교를 끝내기 전까지 결과는
합성 ML 실험, 제어 논리 회귀시험, 가정 비교와 연구 설계에만 사용한다.
