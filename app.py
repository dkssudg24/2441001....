import streamlit as st
import numpy as np
import sympy as sp 
import matplotlib.pyplot as plt

# --- 앱 제목 및 설명 ---
st.title("합성함수 불연속점 시각화 앱")
st.write("수렴/발산 불연속이 있는 함수 f(x), g(x)를 입력하면 합성함수의 불연속점을 찾아 시각화합니다.")

# --- 사용자로부터 함수 입력 받기 ---
# sympy가 이해할 수 있는 형태의 문자열로 함수를 입력받습니다.
f_expr = st.text_input("f(x) 입력 (예: 1/x, tan(x), Abs(x-2))", "1/x")
g_expr = st.text_input("g(x) 입력 (예: x, x+1, sin(x))", "x")

x = sp.symbols('x')

# --- 문자열을 수학 함수로 변환 ---
# try-except 구문을 사용해 잘못된 수식이 입력되면 에러 메시지를 띄웁니다.
try:
    # Abs() 처럼 첫 글자가 대문자인 함수도 인식하도록 수정
    f = sp.sympify(f_expr, locals={'Abs': sp.Abs}) 
    g = sp.sympify(g_expr)
    # 합성함수 h(x) = f(g(x)) 생성
    h = f.subs(x, g)
except Exception as e:
    st.error(f"수식 입력 오류입니다. 다시 확인해주세요: {e}")
    st.stop() # 오류 발생 시 앱 실행 중지

# --- 그래프를 그리기 위한 수치 함수로 변환 ---
# sympy 함수를 numpy가 계산할 수 있는 형태로 바꿔줍니다.
f_lambd = sp.lambdify(x, f, modules=['numpy'])
g_lambd = sp.lambdify(x, g, modules=['numpy'])
h_lambd = sp.lambdify(x, h, modules=['numpy'])

# --- 그래프를 그릴 x값 범위 설정 ---
X = np.linspace(-10, 10, 2000) # 점 개수를 늘려 더 정밀하게

# --- y값 계산 (오류 무시) ---
# 1/0 같은 계산 오류는 잠시 무시하고 진행합니다.
with np.errstate(divide='ignore', invalid='ignore'):
    Yf = f_lambd(X)
    Yg = g_lambd(X)
    Yh = h_lambd(X)

# --- 불연속점 탐지 함수 ---
# y값이 갑자기 크게 변하거나, 무한대/정의되지 않는 지점을 찾습니다.
def find_discontinuities(Y, X):
    disc_points = []
    # np.diff()를 사용해 y값의 변화량을 계산
    diff_Y = np.abs(np.diff(Y)) 
    # 임계값을 설정하여 변화량이 매우 큰 지점을 찾음
    threshold = 15 # 이 값은 조정 가능
    discontinuity_indices = np.where(diff_Y > threshold)[0]

    # nan/inf 값도 불연속점으로 추가
    nan_inf_indices = np.where(np.isnan(Y) | np.isinf(Y))[0]
    
    # 두 리스트를 합치고 중복 제거
    all_indices = np.unique(np.concatenate((discontinuity_indices, nan_inf_indices)))

    for i in all_indices:
        # x좌표 범위 내에 있는 경우에만 추가
        if i < len(X):
            disc_points.append(X[i])

    return disc_points

disc_points = find_discontinuities(Yh, X)

# --- 그래프 그리기 ---
st.write("---")
st.subheader("함수 그래프")
fig, ax = plt.subplots(figsize=(10, 6))

# y축 범위 제한 (너무 큰 값 때문에 그래프가 찌그러지는 것 방지)
ax.set_ylim(-10, 10)

ax.plot(X, Yh, label='h(x) = f(g(x))')
ax.plot(X, Yf, ':', alpha=0.7, label='f(x)')
ax.plot(X, Yg, ':', alpha=0.7, label='g(x)')

# 불연속 지점을 빨간 점선으로 표시
for pt in disc_points:
    ax.axvline(pt, color='red', linestyle='--', linewidth=1, alpha=0.6)

ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- 불연속점 목록 출력 ---
st.write("---")
st.subheader("불연속점 분석 결과")
if disc_points:
    # 중복될 수 있는 점들을 소수점 둘째자리까지 반올림하여 유니크하게 만듦
    unique_points = np.unique(np.round(disc_points, 2))
    st.write("감지된 불연속점 (근사치):", unique_points)
else:
    st.write("지정된 범위 내에서 불연속점이 감지되지 않았습니다.")
