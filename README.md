# 🌳**나무 생존 예측을 통한 도시 수목원 최적화 전략 수립**🌳
<div align="center">
<img src="./data/dataset-cover.png" width="50%" height="50%" alt="Tree">
</div>

---

## 👥 **팀 소개**

<div align="center">

### **🌿 가지가지팀 (SKN_19-mini-3Team) 🌿**

<table>
  <tr>
    <td align="center" width="200px">
      <img src="./data/team/junyung.jpg" width="120px" height="120px" alt="박준영" style="border-radius: 50%;"><br/>
      <b>박준영</b>
    </td>
    <td align="center" width="200px">
      <img src="./data/team/heungjae.jpg" width="120px" height="120px" alt="오흥재" style="border-radius: 50%;"><br/>
      <b>오흥재</b>
    </td>
    <td align="center" width="200px">
      <img src="./data/team/sangmin.jpg" width="120px" height="120px" alt="임상민" style="border-radius: 50%;"><br/>
      <b>임상민</b>
    </td>
    <td align="center" width="200px">
      <img src="./data/team/hyojung.jpg" width="120px" height="120px" alt="장효정" style="border-radius: 50%;"><br/>
      <b>장효정</b>
    </td>
  </tr>
  <tr>
    <td align="center" width="200px">
      <a href="mailto:deneb784@gmail.com"> GitHub</a>
    </td>
    <td align="center" width="200px">
      <a href="mailto:vfxpedia1987@kakao.com"> GitHub</a>
    </td>
    <td align="center" width="200px">
      <a href="mailto:colaa222@gmail.com"> GitHub</a>
    </td>
    <td align="center" width="200px">
      <a href="mailto:koulsh3663@gmail.com"> GitHub</a>
    </td>
  </tr>
</table>

</div>

---

## 📌 우리가 전달하고자 하는 메시지

**"우리는 데이터 분석을 통해 나무 생존 요인을 규명하여, 지속 가능한 도시 수목원 조성과 효과적인 도시환경 개선 전략을 제안한다."**

---

## 📌 프로젝트 목적 (Why)

1. **환경적 배경**

   * AI와 산업 발전으로 인한 환경 문제, [기후 변화 심화](https://apps.climate.copernicus.eu/global-temperature-trend-monitor/?tab=plot)
   * 서울처럼 인구 밀집이 심한 도시의 환경 오염·삶의 질 저하 문제 
   * 해외의 성공적인 도시 녹지화 사례에서 영감
   * [싱가포르 : 도시의 기후 회복력 위한 녹지·수변 공간 조성 계획'](https://www.si.re.kr/bbs/view.do?key=2024100040&pstSn=2309150007)
<div align="center">

  <img src="./data/Temperature%20trend.png" width="80%" height="80%" alt="지구 온난화 추세" />

  <br/><br/>

  <p style="font-size:1.1rem; font-weight:bold; line-height:1.6;">
    사람 체온이 36.5℃에서 38℃로 1.5℃ 오르면 <u>'조금 오른 것'</u>이 아니라 <b>고열·질병 상태</b>입니다.<br/>
    <b>지구도 마찬가지로, +1.5℃는 '아픈 상태'라는 경고입니다.</b>
  </p>

</div>

2. **문제 인식**

   * [도심 속 녹지 부족](https://www.kunews.ac.kr/news/articleView.html?idxno=42843#:~:text=%ED%9C%B4%EC%8B%9D%C2%B7%EB%8C%80%ED%99%94%20%EA%B3%B5%EA%B0%84%20%EC%97%86%EC%96%B4%20%EC%B9%B4%ED%8E%98%EB%A1%9C%E2%80%9D%ED%95%9C%EA%B0%95%EA%B3%B5%EC%9B%90%EA%B3%BC%20%EA%B1%B0%EC%A3%BC%EC%A7%80%EC%97%AD%20%EA%B0%80%EB%A5%B8%20%EB%8F%84%EB%A1%9C%EB%93%A4%EC%84%A0%ED%98%95%EA%B3%B5%EC%9B%90%2C,%EC%84%9C%EC%9A%B8%EC%8B%9C%2025%EA%B0%9C%20%EC%9E%90%EC%B9%98%EA%B5%AC%20%EC%A4%91%203%EB%B2%88%EC%A7%B8%EB%A1%9C%20%EB%85%B9%EC%A7%80%EC%9C%A8%EC%9D%B4%20%EB%82%AE%EB%8B%A4.) → '쉼'의 공간 부족
   * 초기 수목 생존 실패로 인한 경제적 손실 및 관리 효율 저하
<br></br>
<div align="center">
<img src="./data/Problem_01.png" width="50%" height="50%" alt="도심 속 녹지 부족">
</div>
<div align="center">
<img src="./data/Problem_02.png" width="50%" height="50%" alt="도심 속 녹지 부족">
</div>

3. **목적**

   * 데적

<br></br>
<div align="center">
<table>
  <tr>
    <td align="center" width="50%" style="vertical-align: top; padding: 10px;">
      <img src="./data/visual_print/04_show_chemical_relation_scatter_01.png" 
           style="max-width: 100%; height: 300px; width: auto; object-fit: contain;"
           alt="Lignin vs Phenolics 상관관계">
    </td>
    <td align="center" width="50%" style="vertical-align: top; padding: 10px;">
      <img src="./data/visual_print/04_show_chemical_relation_scatter_02.png" 
           style="max-width: 100%; height: 300px; width: auto; object-fit: contain;"
           alt="NSC vs Phenolics 상관관계">
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Lignin vs Phenolics 상관관계</em>
    </td>
    <td align="center">
      <em>NSC vs Phenolics 상관관계</em>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%" style="vertical-align: top; padding: 10px;">
      <img src="./data/visual_print/04_show_chemical_relation_scatter_03.png" 
           style="max-width: 100%; height: 300px; width: auto; object-fit: contain;"
           alt="Lignin vs NSC 상관관계">
    </td>
    <td align="center" width="50%" style="vertical-align: top; padding: 10px;">
      <img src="./data/visual_print/07_show_chemical_histogram.png" 
           style="max-width: 100%; height: 300px; width: auto; object-fit: contain;"
           alt="Phenolics 농도 분포">
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Lignin vs NSC 상관관계</em>
    </td>
    <td align="center">
      <em>Phenolics 농도 분포</em>
    </td>
  </tr>
</table>
</div>

**🧪 화학 성분 영향 분석**
- **Phenolics 고농도**: 34.0% vs 저농도 1.3%
- **Lignin 고농도**: 33.9% vs 저농도 1.5%  
- **NSC 고농도**: 31.5% vs 저농도 3.8%
- **📈 핵심 발견**: 모든 화학 성분에서 고농도가 생존율 크게 향상


<div align="center">
<table>
  <tr>
    <td align="center" style="vertical-align: top; padding: 10px;">
      <img src="./data/visual_print/06_show_survival_ratio.png" 
           style="max-width: 100%; height: 450px; width: auto; object-fit: contain;"
           alt="균근 처리별 생존율">
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>균근 처리별 생존율</em>
    </td>
  </tr>
</table>
</div>

**🍄 균근 처리 효과 분석**
- **균근 미처리**: 210/1,500 (14.0%)
- **균근 처리**: 494/1,283 (38.5%)
- **📈 핵심 발견**: 균근 처리가 생존율을 **2.75배 향상**시킴


**🔬 균근균 타입별 효과**
- **AMF 높음(>20)**: 228/1,112 (20.5%)
- **EMF 높음(>20)**: 268/875 (30.6%)
- **📈 핵심 발견**: EMF가 AMF보다 **1.5배 효과적**


<div align="center">
<table>
  <tr>
    <td align="center" style="vertical-align: top; padding: 10px;">
      <img src="./data/visual_print/05_show_alive_about_time.png" 
           style="max-width: 100%; height: 450px; width: auto; object-fit: contain;"
           alt="시간별 생존율 추이">
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>시간별 생존율 추이</em>
    </td>
  </tr>
</table>
</div>

**⏰ 시간대별 생존 패턴**
- 초기~중기(~90일): 0% (모두 사망 상태)
- **장기(90일~)**: 491/494 (99.4%)
- 📊 **인사이트**: 90일 이후 관찰된 나무들은 대부분 생존


#### **5️⃣ 데이터 정제 및 전처리**

**DataModify 클래스 - 데이터 정제 및 모델링 준비**
```python  
# 데이터 정제 및 분할
├── data_anomaly_edit()      # 이상치 탐지 및 제거 (Alive, Phenolics, NSC, Lignin)
├── split_feature_label()    # Feature-Label 분리 및 Train/Test 분할
└── scale_data()             # 연속형 변수 표준화 (StandardScaler)
```

**DataVisualize 클래스 - 통계적 시각화**
```python
# 탐색적 데이터 분석 (EDA)
├── show_boxplot_for_continuous_value()           # 연속형 변수 분포 시각화
├── show_anomaly_score()                          # Isolation Forest 이상치 스코어링
├── show_heatmap_for_continuous_value()           # 변수간 상관관계 히트맵
├── show_survival_ratio()                         # 범주별 생존율 분석
├── show_survival_ratio_with_threshold()          # 임계값 기반 생존율 분석  
├── show_alive_about_time()                       # 시간별 생존율 추이
├── show_survival_heatmap_by_soil()               # 토양-종별 생존율 매트릭스
├── show_chemical_histogram()                     # 화학성분 분포 히스토그램
├── show_chemical_relation_scatter()              # 화학성분 간 상관관계
├── show_chemical_relation_by_survive()           # 생존여부별 화학성분 분석
└── show_hitmap_by_Myco()                         # 균근 처리 교차분석
```

---

## 🎯 **핵심 인사이트 및 도출된 전략**

### 💡 **주요 발견 사항**

1. **🍄 균근 처리의 유의미한 효과**
   - 균근 처리 시 생존율 **14.0% → 38.5%** (2.8배 향상)
   - EMF 균근균이 AMF보다 1.5배 효과적 (30.6% vs 20.5%)

2. **🧪 화학 성분의 결정적 역할**
   - Phenolics 고농도: 37.3% vs 저농도 17.6% (2.1배 향상)
   - 식물 내 방어 화합물이 생존에 핵심적 역할

3. **⏰ 실험 설계의 특성**
   - 90일 이상: 실험 목적상 의도적 수확으로 실험 종료
   - 초기 90일이 생존 결정의 **Critical Period**

### 🏢 **도시 수목원 조성 전략**

#### **📋 최적 식재 가이드라인**
1. **필수 조건**: 균근 처리 (생존율 2.8배 향상)
2. **균근균**: EMF(외생균근균) 접종 우선 (AMF보다 1.5배 효과적)
3. **관리 집중**: 식재 후 90일간 집중 관리

#### **💰 경제적 효과**
- 기존 생존율 14.0% → 개선 후 **34.4%** (2.5배 향상)
- 균근 처리를 통한 대폭적인 생존율 향상으로 재조림 비용 절감 효과 기대
- 정확한 경제적 효과는 추가적인 비용-편익 분석 필요

#### **🌱 지속가능한 도시 생태계 구축**
- 데이터 기반 과학적 식재로 **건강한 쉼터 공간** 제공
- 초기 90일 집중 관리를 통한 **효율적 자원 배분**
- 시민 삶의 질 향상과 **환경 개선 동시 달성**

---


## 📚 **자료**
>
> [[이미지] 코페르니쿠스 기후 변화 서비스](https://apps.climate.copernicus.eu/global-temperature-trend-monitor/?tab=plot)
> 
> [[데이터] 코페르니쿠스 기후 변화 서비스](https://climate.copernicus.eu/)
>
> [[데이터] 서울연구원 - 세계도시정책동향](https://www.si.re.kr/bbs/view.do?key=2024100040&pstSn=2309150007)
>
> [[데이터] Kaggle.com/datasets](https://www.kaggle.com/datasets/yekenot/tree-survival-prediction/data)
>
> [[기사] 서울 도심 녹지율 3.7%](https://www.lafent.com/inews/news_view.html?news_id=130618&mcd=A01&page=60)
>
> [[기사] 도시민을 위한 그린 인프라 부족](https://www.kunews.ac.kr/news/articleView.html?idxno=42843)
