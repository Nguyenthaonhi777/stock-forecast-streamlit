import os
import random
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential


# --------------------------- App configuration ---------------------------
st.set_page_config(
    page_title="Dự báo giá cổ phiếu | LSTM",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

SEED = 42
WINDOW_SIZE = 60
DEFAULT_START_DATE = date(2018, 1, 1)

STOCK_OPTIONS = {
    "Vinamilk (VNM)": "VNM.VN",
    "FPT (FPT)": "FPT.VN",
    "Hòa Phát (HPG)": "HPG.VN",
    "Thế Giới Di Động (MWG)": "MWG.VN",
    "Vingroup (VIC)": "VIC.VN",
    "Vietcombank (VCB)": "VCB.VN",
    "SSI (SSI)": "SSI.VN",
    "VN-Index": "^VNINDEX",
    "Nhập mã khác...": "__custom__",
}


def set_seed(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


@st.cache_data(ttl=3600, show_spinner=False)
def download_stock_data(ticker: str, start_date: date) -> pd.DataFrame:
    """Download adjusted daily prices; cached for one hour."""
    end_date = date.today() + timedelta(days=1)  # yfinance end date is exclusive
    df = yf.download(
        ticker,
        start=start_date.isoformat(),
        end=end_date.isoformat(),
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return pd.DataFrame()

    # Accommodate yfinance versions that return MultiIndex columns.
    if isinstance(df.columns, pd.MultiIndex):
        close_candidates = [col for col in df.columns if "Close" in col]
        if close_candidates:
            close = df[close_candidates[0]]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
        else:
            return pd.DataFrame()
    elif "Close" in df.columns:
        close = df["Close"]
    else:
        return pd.DataFrame()

    close = pd.to_numeric(close, errors="coerce").dropna()
    result = close.to_frame(name="Close")
    result.index = pd.to_datetime(result.index).tz_localize(None)
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result


def create_sequences(values: np.ndarray, window_size: int = WINDOW_SIZE):
    X, y = [], []
    for i in range(window_size, len(values)):
        X.append(values[i - window_size : i])
        y.append(values[i])
    return np.asarray(X), np.asarray(y)


def build_model() -> Sequential:
    model = Sequential(
        [
            LSTM(50, return_sequences=True, input_shape=(WINDOW_SIZE, 1)),
            LSTM(50),
            Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def run_forecast(
    df: pd.DataFrame,
    ticker: str,
    forecast_days: int,
    epochs: int,
):
    if len(df) <= WINDOW_SIZE * 3:
        raise ValueError("Dữ liệu quá ít. Hãy chọn mã có ít nhất khoảng 180 phiên dữ liệu.")

    set_seed()
    data = df[["Close"]].values.astype(float)
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    X_train, y_train = create_sequences(train_scaled)
    test_combined = np.vstack((train_scaled[-WINDOW_SIZE:], test_scaled))
    X_test, y_test = create_sequences(test_combined)

    if len(X_train) < 10 or len(X_test) < 2:
        raise ValueError("Không đủ dữ liệu để huấn luyện và đánh giá mô hình.")

    val_size = max(1, int(0.1 * len(X_train)))
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train_final, y_train_final = X_train[:-val_size], y_train[:-val_size]

    tf.keras.backend.clear_session()
    model = build_model()
    early_stop = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    history = model.fit(
        X_train_final,
        y_train_final,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        shuffle=False,
        callbacks=[early_stop],
        verbose=0,
    )

    test_pred_scaled = model.predict(X_test, verbose=0)
    test_pred = scaler.inverse_transform(test_pred_scaled).ravel()
    y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_test_real, test_pred)))
    mae = float(mean_absolute_error(y_test_real, test_pred))
    r2 = float(r2_score(y_test_real, test_pred))

    latest_60 = scaler.transform(data)[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
    future_scaled = []
    sequence = latest_60.copy()
    for _ in range(forecast_days):
        next_pred = model.predict(sequence, verbose=0)
        next_value = float(next_pred[0, 0])
        future_scaled.append(next_value)
        sequence = np.concatenate(
            [sequence[:, 1:, :], np.array(next_value).reshape(1, 1, 1)], axis=1
        )

    future_values = scaler.inverse_transform(
        np.array(future_scaled).reshape(-1, 1)
    ).ravel()

    latest_date = pd.Timestamp(df.index[-1]).normalize()
    # Weekdays only; exchange holidays are not included in this simple date estimate.
    future_dates = pd.bdate_range(
        start=latest_date + pd.Timedelta(days=1), periods=forecast_days
    )
    forecast_df = pd.DataFrame(
        {
            "Ngày dự kiến": future_dates,
            "Giá dự báo": future_values,
        }
    )
    forecast_df["Ngày dự kiến"] = forecast_df["Ngày dự kiến"].dt.strftime("%d/%m/%Y")

    return {
        "ticker": ticker,
        "data": df,
        "forecast": forecast_df,
        "future_values": future_values,
        "future_dates": future_dates,
        "latest_close": float(data[-1, 0]),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "test_actual": y_test_real,
        "test_pred": test_pred,
        "history": history.history,
        "epochs_ran": len(history.history.get("loss", [])),
    }


def format_price(value: float) -> str:
    return f"{value:,.2f}"


# ------------------------------- Page design ------------------------------
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 2.5rem;}
      .hero {
        padding: 1.35rem 1.5rem; border-radius: 16px;
        background: linear-gradient(115deg, #102a43 0%, #176b87 58%, #2c9c91 100%);
        color: white; margin-bottom: 1rem;
      }
      .hero h1 {color: white; font-size: 2rem; margin: 0 0 .35rem 0;}
      .hero p {color: #edf7fa; margin: 0; font-size: 1rem;}
      .small-note {color: #627d98; font-size: .88rem;}
    </style>
    <div class="hero">
      <h1>📈 Dự báo giá cổ phiếu bằng LSTM</h1>
      <p>Chọn mã cổ phiếu, nhấn nút dự báo và xem ước tính giá cho các phiên tiếp theo.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Thiết lập dự báo")
    with st.form("forecast_form"):
        stock_label = st.selectbox("Mã cổ phiếu", list(STOCK_OPTIONS.keys()), index=0)
        custom_ticker = ""
        if stock_label == "Nhập mã khác...":
            custom_ticker = st.text_input(
                "Mã Yahoo Finance",
                value="",
                placeholder="Ví dụ: AAPL hoặc FPT.VN",
                help="Nhập mã đúng theo Yahoo Finance.",
            )
        start_date = st.date_input(
            "Ngày bắt đầu lấy dữ liệu",
            value=DEFAULT_START_DATE,
            min_value=date(2000, 1, 1),
            max_value=date.today(),
        )
        forecast_days = st.slider("Số phiên cần dự báo", min_value=1, max_value=30, value=10)
        epochs = st.slider("Số vòng huấn luyện tối đa", min_value=5, max_value=30, value=10)
        submit = st.form_submit_button("🚀 DỰ BÁO", type="primary", use_container_width=True)

    st.markdown("---")
    st.caption("Mô hình: LSTM hai lớp • Dữ liệu: giá đóng cửa đã điều chỉnh • Cửa sổ đầu vào: 60 phiên")

if submit:
    selected_ticker = (
        custom_ticker.strip().upper()
        if stock_label == "Nhập mã khác..."
        else STOCK_OPTIONS[stock_label]
    )
    if not selected_ticker:
        st.error("Vui lòng nhập mã cổ phiếu trước khi dự báo.")
        st.session_state.pop("forecast_result", None)
    else:
        try:
            with st.spinner(f"Đang tải dữ liệu và huấn luyện mô hình cho {selected_ticker}... Vui lòng chờ đến khi có kết quả."):
                stock_df = download_stock_data(selected_ticker, start_date)
                if stock_df.empty:
                    raise ValueError(
                        "Không tìm thấy dữ liệu giá. Kiểm tra mã cổ phiếu hoặc thử lại sau."
                    )
                result = run_forecast(stock_df, selected_ticker, forecast_days, epochs)
                result["stock_label"] = stock_label
                result["start_date"] = start_date.isoformat()
                result["requested_days"] = forecast_days
                st.session_state["forecast_result"] = result
                st.session_state.pop("forecast_error", None)
        except Exception as exc:
            st.session_state.pop("forecast_result", None)
            st.session_state["forecast_error"] = str(exc)

if st.session_state.get("forecast_error"):
    st.error(f"Không thể tạo dự báo: {st.session_state['forecast_error']}")

result = st.session_state.get("forecast_result")
if result:
    st.subheader(f"Kết quả dự báo: {result['ticker']}")
    st.caption(
        f"Dữ liệu từ {pd.Timestamp(result['data'].index[0]).strftime('%d/%m/%Y')} "
        f"đến {pd.Timestamp(result['data'].index[-1]).strftime('%d/%m/%Y')} • "
        f"Huấn luyện {result['epochs_ran']} vòng • Dự báo {result['requested_days']} phiên"
    )

    latest_close = result["latest_close"]
    final_forecast = float(result["future_values"][-1])
    absolute_change = final_forecast - latest_close
    pct_change = (absolute_change / latest_close * 100) if latest_close else 0.0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Giá đóng cửa gần nhất", format_price(latest_close))
    col2.metric(
        f"Giá dự báo phiên thứ {result['requested_days']}",
        format_price(final_forecast),
        f"{pct_change:+.2f}% so với giá gần nhất",
    )
    col3.metric("RMSE (tập kiểm tra)", format_price(result["rmse"]))
    col4.metric("MAE (tập kiểm tra)", format_price(result["mae"]))

    forecast_tab, accuracy_tab, training_tab = st.tabs(
        ["🔮 Dự báo", "📊 Đánh giá mô hình", "🧠 Quá trình huấn luyện"]
    )

    with forecast_tab:
        left, right = st.columns([1.6, 1])
        with left:
            st.markdown("#### Giá lịch sử và giá dự báo")
            hist = result["data"].tail(180)
            fig, ax = plt.subplots(figsize=(10, 4.8))
            ax.plot(hist.index, hist["Close"], label="Giá đóng cửa lịch sử", linewidth=1.8)
            ax.plot(
                result["future_dates"],
                result["future_values"],
                marker="o",
                markersize=4,
                linestyle="--",
                label="Giá dự báo LSTM",
                linewidth=1.8,
            )
            ax.set_xlabel("Ngày")
            ax.set_ylabel("Giá")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.autofmt_xdate()
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with right:
            st.markdown("#### Bảng giá dự báo")
            display_forecast = result["forecast"].copy()
            display_forecast["Giá dự báo"] = display_forecast["Giá dự báo"].map(format_price)
            st.dataframe(display_forecast, hide_index=True, use_container_width=True)
            csv = result["forecast"].to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Tải kết quả CSV",
                data=csv,
                file_name=f"du_bao_{result['ticker'].replace('^', '')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.markdown(
                '<p class="small-note">Ngày dự kiến chỉ bỏ qua thứ Bảy và Chủ nhật, chưa hiệu chỉnh theo ngày nghỉ lễ của từng sàn.</p>',
                unsafe_allow_html=True,
            )

    with accuracy_tab:
        st.markdown("#### Giá thực tế và giá mô hình dự đoán trên tập kiểm tra")
        fig, ax = plt.subplots(figsize=(11, 4.8))
        ax.plot(result["test_actual"], label="Giá thực tế", linewidth=1.5)
        ax.plot(result["test_pred"], label="Giá dự đoán", linewidth=1.5)
        ax.set_xlabel("Phiên trong tập kiểm tra")
        ax.set_ylabel("Giá")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSE", format_price(result["rmse"]))
        m2.metric("MAE", format_price(result["mae"]))
        m3.metric("R²", f"{result['r2']:.4f}")
        st.caption("RMSE và MAE càng thấp thường càng tốt. R² gần 1 thể hiện mức độ khớp tốt hơn trên tập kiểm tra; kết quả âm có thể xảy ra.")

    with training_tab:
        st.markdown("#### Loss trong quá trình huấn luyện")
        history = result["history"]
        fig, ax = plt.subplots(figsize=(10, 4.2))
        ax.plot(history.get("loss", []), label="Train Loss")
        ax.plot(history.get("val_loss", []), label="Validation Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Squared Error")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.warning(
        "Lưu ý: Đây là kết quả ước tính từ mô hình LSTM chỉ sử dụng dữ liệu giá đóng cửa trong quá khứ. "
        "Mô hình không xét đầy đủ tin tức, báo cáo tài chính, thanh khoản, sự kiện doanh nghiệp hoặc yếu tố vĩ mô. "
        "Dự báo có thể sai lệch và không phải khuyến nghị mua/bán cổ phiếu."
    )
else:
    st.info("Chọn mã cổ phiếu và các thông số ở thanh bên, sau đó bấm **DỰ BÁO** để tạo kết quả.")
    st.markdown("### Ứng dụng cung cấp")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**📈 Dự báo giá**\n\nƯớc tính giá đóng cửa cho 1–30 phiên tiếp theo.")
    c2.markdown("**📊 Đánh giá mô hình**\n\nXem RMSE, MAE, R² và so sánh giá thực tế với giá dự đoán.")
    c3.markdown("**📥 Xuất dữ liệu**\n\nTải bảng giá dự báo về máy dưới dạng CSV.")
