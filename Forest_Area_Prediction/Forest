import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import koreanize_matplotlib
import random

# 데이터 로드
df = pd.read_csv('지역별_연도별_산림면적.csv')

# 지역명 추출 및 중복 제거
arr = df['지역명']
result = list(arr.unique())

# 지역명 선택 안내
print('2050년 산림면적을 알아 보고 싶은 지역을 아래에서 고르세요.')
print(', '.join(result), '\n')

# 지역명 입력
while True:
    location = input('알아 보고 싶은 지역명을 입력해 주세요 : ')
    if location not in result:
        print('\n존재하지 않는 지역명 입니다. 다시 입력해주세요.\n')
    else:
        break

# 선택된 지역 데이터 필터링
df = df[df["지역명"] == location]

# 데이터 형 변환
df = df.astype({'측정연도':'int'})

# 데이터프레임에서 특정 값 추출
last_mountain_area = df.iloc[-1, -1]

# 연도별로 데이터를 정렬
df = df.sort_values('측정연도')
# 모든 연도를 포함하는 DataFrame 생성
all_years = pd.DataFrame({'측정연도': np.arange(df['측정연도'].min(), df['측정연도'].max() + 1)})
# 원본 데이터프레임과 병합하여 모든 연도를 포함하도록 함
df = pd.merge(all_years, df, on='측정연도', how='left')
# 선형 보간법을 사용하여 결측치를 채움
df['면적'] = df['면적'].interpolate()

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(df['측정연도'], df['면적'], color="skyblue", alpha=0.5, marker='o', markerfacecolor='blue', markersize=8)
plt.xticks(np.arange(df['측정연도'].min(), df['측정연도'].max() + 1, 1), rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('측정 연도(년)', labelpad=10, loc='right', color='blue', size=12)
plt.ylabel('산림면적(ha)', labelpad=10, loc='top', rotation=0, color='red', size=12)
plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f'))
plt.grid(alpha=0.3)
plt.title(f"{location}의 산림면적 현황 그래프", size=22)
plt.tight_layout()
plt.show()

# 인덱스(측정연도)를 2차원 배열로 변환
years = df['측정연도'].values.reshape(-1, 1)
# '면적' 열의 값을 배열로 정의
areas = df["면적"].values

# 주어진 데이터를 훈련 데이터와 테스트 데이터로 나눔
X_train, X_test, y_train, y_test = train_test_split(years, areas, test_size=0.2, random_state=42)

# 선형 회귀 모델 객체를 만듦
model = LinearRegression()
# X_train, y_train 으로 모델에 학습
model.fit(X_train, y_train)

# 2021년부터 2050년까지의 연도를 배열로 만듦
future_years = np.arange(2021, 2051).reshape(-1, 1)
# 미래 연도의 면적을 예측
predicted_areas = model.predict(future_years)

# 2021년부터 2050년까지 예측된 지역의 면적을 소수점 둘째 자리까지 출력
for year, area in zip(range(2021, 2051), predicted_areas):
    print('\033[97m' + f"예측된 {location}의 {year}년 면적 :" + '\033[0m' + '\033[96m' + f"{area: .2f}ha" + '\033[0m')

# 예측 결과를 데이터 프레임으로 변환
df = pd.DataFrame({"측정 연도": np.arange(2021, 2051), "면적": predicted_areas})

# 데이터 프레임 인덱스 설정
df.set_index('측정 연도', inplace=True)

# 각 지역에 따른 시각화의 Y축 눈금 및 범위 설정
if location == '서울특별시':
    interval = 1000
    min_deviation = 726
    max_deviation = 1225
elif location == '부산광역시':
    interval = 1000
    min_deviation = 1031
    max_deviation = 1000
elif location == '대구광역시':
    interval = 1000
    min_deviation = 1164
    max_deviation = 2000
elif location == '인천광역시':
    interval = 1000
    min_deviation = 805
    max_deviation = 1000
elif location == '광주광역시':
    interval = 1000
    min_deviation = 1554
    max_deviation = 1400
elif location == '대전광역시':
    interval = 1000
    min_deviation = 903
    max_deviation = 1000
elif location == '울산광역시':
    interval = 1000
    min_deviation = 1150
    max_deviation = 2000
elif location == '경기도':
    interval = 10000
    min_deviation = 8696
    max_deviation = 6000
elif location == '강원도':
    interval = 100000
    min_deviation = 1079
    max_deviation = 28200
elif location == '충청북도':
    interval = 10000
    min_deviation = 7820
    max_deviation = 8200
elif location == '충청남도':
    interval = 10000
    min_deviation = 17516
    max_deviation = 7600
elif location == '전라북도':
    interval = 10000
    min_deviation = 9191
    max_deviation = 7000
elif location == '전라남도':
    interval = 10000
    min_deviation = 8921
    max_deviation = 1000
elif location == '경상북도':
    interval = 10000
    min_deviation = 6230
    max_deviation = 7000
elif location == '경상남도':
    interval = 10000
    min_deviation = 7801
    max_deviation = 10050
elif location == '제주도':
    interval = 1000
    min_deviation = 93
    max_deviation = 200

# 선택한 지역의 2050년까지 예측한 면적의 가장 높은 면적, 가장 낮은 면적, 평균 면적을 각 변수에 대입
areaMax = round(df['면적'].max(), 2)
areaMin = round(df['면적'].min(), 2)
areaMean = round(df['면적'].mean(), 2)

# 그래프 시각화
plt.figure(figsize=(10, 6))
df["면적"].plot(color="skyblue", alpha=0.5, marker='o', markerfacecolor='blue', markersize=6)
plt.xticks(np.arange(2021, 2051).tolist(), rotation=45, fontsize=12)
plt.yticks(np.arange(areaMin - min_deviation, areaMax + max_deviation, interval), fontsize=12)
plt.xlabel('측정 연도(년)', labelpad=10, loc='right', color='blue', size=12)
plt.ylabel('면적(ha)', labelpad=10, loc='top', rotation=0, color='red', size=12)
plt.grid(alpha=0.3)
plt.title(f"{location}의 미래 산림면적 예측 그래프", size=22)
plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.0f'))
plt.legend()
plt.tight_layout()
plt.show()

# 면적이 가장 높았던 연도, 가장 낮았던 연도를 각 변수에 대입
areaMaxYear = df.index[df['면적'] == df['면적'].max()].tolist()
areaMinYear = df.index[df['면적'] == df['면적'].min()].tolist()

# 최소 면적일 때 면적의 범위에 따른 육상 생태계 변화 데이터
if areaMin <= 13000:
    print('\033[91m' + '※ [경고] 탄소 배출량 증가로 인한 기온 변화 악화 가능성 증가 ※' + '\033[0m')

# 2021년 ~ 2050년까지 지역의 가장 높은 면적, 가장 낮은 면적, 평균 면적을 출력
print('\033[93m' + f'[2021년 ~ 2050년] {location}의 가장 높은 면적 / 가장 낮은 면적 / 평균 면적 :' + '\033[0m')
print('\033[97m' + f'{location}의 가장 높은 면적은 {areaMaxYear[0]}년도 ' + '\033[0m' + '\033[96m' + f'{areaMax}ha' + '\033[97m' + "입니다.")
print('\033[97m' + f'{location}의 가장 낮은 면적은 {areaMinYear[0]}년도 ' + '\033[0m' + '\033[96m' + f'{areaMin}ha' + '\033[97m' + " 입니다.")
print('\033[97m' + f'{location}의 평균 면적은 ' + '\033[0m' + '\033[96m' + f'{areaMax}ha' + '\033[97m' + " 입니다.")