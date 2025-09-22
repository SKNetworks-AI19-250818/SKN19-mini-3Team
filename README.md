<div align="center">
<img src="./data/dataset-cover.png" width="50%" height="50%" alt="Tree">
</div>

## 📌 우리가 전달하고자 하는 메시지

**“우리는 데이터 분석을 통해 나무 생존 요인을 규명하여, 지속 가능한 도시 수목원 조성과 효과적인 도시환경 개선 전략을 제안한다.”**

---

## 📌 프로젝트 목적 (Why)

1. **환경적 배경**

   * AI와 산업 발전으로 인한 환경 문제, [기후 변화 심화](https://apps.climate.copernicus.eu/global-temperature-trend-monitor/?tab=plot)
   * 서울처럼 인구 밀집이 심한 도시의 환경 오염·삶의 질 저하 문제 
   * 해외의 성공적인 도시 녹지화 사례에서 영감
   * [싱가포르 : 도시의 기후 회복력 위한 녹지·수변 공간 조성 계획’](https://www.si.re.kr/bbs/view.do?key=2024100040&pstSn=2309150007)
<div align="center"> 
<img src="./data/Temperature%20trend.png" width="80%" height="80%" alt="지구 온난화">
</div>

2. **문제 인식**

   * [도심 속 녹지 부족](https://www.kunews.ac.kr/news/articleView.html?idxno=42843#:~:text=%ED%9C%B4%EC%8B%9D%C2%B7%EB%8C%80%ED%99%94%20%EA%B3%B5%EA%B0%84%20%EC%97%86%EC%96%B4%20%EC%B9%B4%ED%8E%98%EB%A1%9C%E2%80%9D%ED%95%9C%EA%B0%95%EA%B3%B5%EC%9B%90%EA%B3%BC%20%EA%B1%B0%EC%A3%BC%EC%A7%80%EC%97%AD%20%EA%B0%80%EB%A5%B8%20%EB%8F%84%EB%A1%9C%EB%93%A4%EC%84%A0%ED%98%95%EA%B3%B5%EC%9B%90%2C,%EC%84%9C%EC%9A%B8%EC%8B%9C%2025%EA%B0%9C%20%EC%9E%90%EC%B9%98%EA%B5%AC%20%EC%A4%91%203%EB%B2%88%EC%A7%B8%EB%A1%9C%20%EB%85%B9%EC%A7%80%EC%9C%A8%EC%9D%B4%20%EB%82%AE%EB%8B%A4.) → ‘쉼’의 공간 부족
   * 초기 수목 생존 실패로 인한 경제적 손실 및 관리 효율 저하
<div align="center">
<img src="./data/Problem_01.png" width="50%" height="50%" alt="도심 속 녹지 부족">
</div>
<div align="center">
<img src="./data/Problem_02.png" width="50%" height="50%" alt="도심 속 녹지 부족">
</div>

3. **목적**

   * 데이터 기반으로 **어떤 나무가 어떤 조건에서 잘 자라는지** 과학적으로 규명
   * 도시 수목원·녹지 공간 조성을 위한 **최적 입지 및 관리 전략 제시**
   * 결과적으로, 도시민에게 **건강한 쉼터 제공 + 지속 가능한 도시 생태계 구축**

---

## 📌 우리는 데이터를 어떻게 활용할 것인가? (How)

* [**Tree\_Data.csv**](https://www.kaggle.com/datasets/yekenot/tree-survival-prediction/data)를 분석하여 다음을 도출:

  1. **생존율에 영향을 주는 주요 요인** (토양, 멸균 여부, 균근, 화학 성분 등)
  2. **종별 최적 조건** (예: 어떤 수종은 저광량에서도 잘 자라고, 어떤 종은 특정 토양에서만 생존율이 높음)
  3. **경제적 관리 전략** (초기 묘목의 생존 가능성을 높여 재조림 비용 절감)

* **EDA & 예측 모델링**을 통해:

  * “이 종은 XX 토양+빛 조건에서 가장 잘 생존한다” 같은 **가이드라인 제공**
  * 도심 내 수목원 조성 시 **데이터 기반 의사결정** 지원

---

## 자료
>
> [[이미지] 코페르니쿠스 기후 변화 서비스](https://apps.climate.copernicus.eu/global-temperature-trend-monitor/?tab=plot)
> 
> [[데이터] 코페르니쿠스 기후 변화 서비스](https://climate.copernicus.eu/)
>
>[[데이터] 서울연구원 - 세계도시정책동향](https://www.si.re.kr/bbs/view.do?key=2024100040&pstSn=2309150007)
>
> [[데이터] Kaggel.com/datasets](https://www.kaggle.com/datasets/yekenot/tree-survival-prediction/data)
>
> [[기사] 서울 도심 녹지율 3.7%](https://www.lafent.com/inews/news_view.html?news_id=130618&mcd=A01&page=60)
>
> [[기사] 도시민을 위한 그린 인프라 부족](https://www.kunews.ac.kr/news/articleView.html?idxno=42843#:~:text=%ED%9C%B4%EC%8B%9D%C2%B7%EB%8C%80%ED%99%94%20%EA%B3%B5%EA%B0%84%20%EC%97%86%EC%96%B4%20%EC%B9%B4%ED%8E%98%EB%A1%9C%E2%80%9D%ED%95%9C%EA%B0%95%EA%B3%B5%EC%9B%90%EA%B3%BC%20%EA%B1%B0%EC%A3%BC%EC%A7%80%EC%97%AD%20%EA%B0%80%EB%A5%B8%20%EB%8F%84%EB%A1%9C%EB%93%A4%EC%84%A0%ED%98%95%EA%B3%B5%EC%9B%90%2C,%EC%84%9C%EC%9A%B8%EC%8B%9C%2025%EA%B0%9C%20%EC%9E%90%EC%B9%98%EA%B5%AC%20%EC%A4%91%203%EB%B2%88%EC%A7%B8%EB%A1%9C%20%EB%85%B9%EC%A7%80%EC%9C%A8%EC%9D%B4%20%EB%82%AE%EB%8B%A4.)
>
