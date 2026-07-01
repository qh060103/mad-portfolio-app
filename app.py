import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.optimize import linprog


ASSET_NAMES = ["现金/货币基金", "债券基金", "股票指数基金", "主动权益基金", "黄金ETF"]
PERIODS = [f"第{i}月" for i in range(1, 13)]

# 原始月度收益率数据，单位为百分比。
RETURN_DATA_PERCENT = np.array(
    [
        [0.15, 0.40, 2.50, 3.50, 1.00],
        [0.16, 0.20, -3.00, -5.00, -1.50],
        [0.14, -0.10, 4.00, 6.00, 2.50],
        [0.15, 0.50, 1.00, 2.00, 0.50],
        [0.15, 0.30, -5.00, -7.00, 4.00],
        [0.16, 0.10, 6.00, 8.00, -2.00],
        [0.15, 0.40, -2.00, -3.00, 1.50],
        [0.14, 0.20, 3.00, 4.50, -1.00],
        [0.15, 0.30, 4.00, 5.00, 3.00],
        [0.16, -0.20, -4.00, -6.00, 2.00],
        [0.14, 0.50, 5.00, 7.00, -3.00],
        [0.15, 0.20, -1.00, -2.00, 1.00],
    ],
    dtype=float,
)


def get_return_dataframe() -> pd.DataFrame:
    """返回用于页面展示的收益率数据表。"""
    return pd.DataFrame(RETURN_DATA_PERCENT, columns=ASSET_NAMES, index=PERIODS)


def solve_mad_portfolio(risk_limit_percent: float) -> dict:
    """求解均值-绝对偏差投资组合线性规划模型。

    参数 risk_limit_percent 使用页面输入口径，例如 0.60 表示 0.60%。
    """
    returns = RETURN_DATA_PERCENT / 100
    risk_limit_decimal = risk_limit_percent / 100
    periods_count, asset_count = returns.shape
    mean_returns = returns.mean(axis=0)
    deviations = returns - mean_returns

    # 决策变量顺序：[x_1, ..., x_n, d_1, ..., d_T]。
    variable_count = asset_count + periods_count
    objective = np.concatenate([-mean_returns, np.zeros(periods_count)])

    # 满仓约束：sum(x_i) = 1。
    a_eq = np.zeros((1, variable_count))
    a_eq[0, :asset_count] = 1
    b_eq = np.array([1.0])

    a_ub_rows = []
    b_ub_values = []

    # MAD 风险约束：(1/T) * sum(d_t) <= D。
    mad_row = np.zeros(variable_count)
    mad_row[asset_count:] = 1 / periods_count
    a_ub_rows.append(mad_row)
    b_ub_values.append(risk_limit_decimal)

    # 线性化绝对值：d_t >= 偏离值，d_t >= -偏离值。
    for period_index in range(periods_count):
        positive_row = np.zeros(variable_count)
        positive_row[:asset_count] = deviations[period_index]
        positive_row[asset_count + period_index] = -1
        a_ub_rows.append(positive_row)
        b_ub_values.append(0.0)

        negative_row = np.zeros(variable_count)
        negative_row[:asset_count] = -deviations[period_index]
        negative_row[asset_count + period_index] = -1
        a_ub_rows.append(negative_row)
        b_ub_values.append(0.0)

    bounds = [(0, None)] * variable_count
    result = linprog(
        c=objective,
        A_ub=np.array(a_ub_rows),
        b_ub=np.array(b_ub_values),
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        return {
            "success": False,
            "message": result.message,
            "weights": np.zeros(asset_count),
        }

    weights = result.x[:asset_count]
    portfolio_returns = returns @ weights
    expected_monthly_return = float(mean_returns @ weights)
    annual_return = (1 + expected_monthly_return) ** 12 - 1
    actual_mad = float(np.mean(np.abs(portfolio_returns - expected_monthly_return)))

    return {
        "success": True,
        "weights": weights,
        "expected_monthly_return_percent": expected_monthly_return * 100,
        "expected_annual_return_percent": annual_return * 100,
        "actual_mad_percent": actual_mad * 100,
        "portfolio_returns_percent": portfolio_returns * 100,
    }


def build_allocation_dataframe(weights: np.ndarray) -> pd.DataFrame:
    """整理最优投资比例表。"""
    return pd.DataFrame(
        {
            "资产类别": ASSET_NAMES,
            "最优投资比例": weights,
            "最优投资比例（%）": weights * 100,
        }
    )


def run_sensitivity_analysis(risk_limits_percent: list[float]) -> pd.DataFrame:
    """计算不同 MAD 风险上限下的最优年化收益率。"""
    records = []
    for risk_limit in risk_limits_percent:
        solution = solve_mad_portfolio(risk_limit)
        records.append(
            {
                "MAD风险上限（%）": risk_limit,
                "最优年化收益率（%）": (
                    solution["expected_annual_return_percent"] if solution["success"] else np.nan
                ),
                "是否可行": "可行" if solution["success"] else "不可行",
            }
        )
    return pd.DataFrame(records)


def render_metric_cards(solution: dict) -> None:
    col1, col2, col3 = st.columns(3)
    col1.metric("月度预期收益率", f"{solution['expected_monthly_return_percent']:.4f}%")
    col2.metric("年化预期收益率", f"{solution['expected_annual_return_percent']:.4f}%")
    col3.metric("实际月度 MAD 风险", f"{solution['actual_mad_percent']:.4f}%")


def main() -> None:
    st.set_page_config(
        page_title="基于 MAD 线性规划模型的投资组合优化 App",
        layout="wide",
    )

    st.title("基于 MAD 线性规划模型的投资组合优化 App")
    st.write(
        "本 App 使用均值-绝对偏差（MAD）线性规划模型，在给定月度 MAD 风险上限的条件下，"
        "求解五类资产的最优投资比例，并展示组合收益与风险结果。"
    )

    with st.sidebar:
        st.header("参数输入")
        risk_limit_percent = st.number_input(
            "月度 MAD 风险上限 D（%）",
            min_value=0.0,
            max_value=10.0,
            value=0.60,
            step=0.05,
            format="%.3f",
            help="例如输入 0.60 表示 0.60%，模型计算时会自动转换为小数 0.006。",
        )
        st.caption("D 越大，模型允许的收益波动越高。")

    solution = solve_mad_portfolio(risk_limit_percent)

    if not solution["success"]:
        st.error("当前 MAD 风险上限下模型无可行解，请适当增大 D 后重新计算。")
    else:
        render_metric_cards(solution)

        allocation_df = build_allocation_dataframe(solution["weights"])
        display_df = allocation_df.copy()
        display_df["最优投资比例"] = display_df["最优投资比例"].map("{:.6f}".format)
        display_df["最优投资比例（%）"] = display_df["最优投资比例（%）"].map("{:.4f}%".format)

        st.subheader("最优资产配置表")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        st.subheader("资产配置柱状图")
        chart = px.bar(
            allocation_df,
            x="资产类别",
            y="最优投资比例（%）",
            text=allocation_df["最优投资比例（%）"].map(lambda value: f"{value:.2f}%"),
            color="资产类别",
        )
        chart.update_layout(
            xaxis_title="资产类别",
            yaxis_title="投资比例（%）",
            showlegend=False,
            bargap=0.35,
        )
        chart.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(chart, use_container_width=True)

    st.subheader("模型解释")
    st.markdown(
        r"""
        该模型以组合平均月收益率最大化为目标，在满仓且不允许卖空的条件下，通过辅助变量
        \(d_t\) 将组合收益相对平均收益的绝对偏差线性化。用户输入的 D 是月度 MAD 风险上限，
        页面输入单位为百分比，程序会在求解前转换为小数。若 D 设置过低，可能不存在满足风险
        约束的资产组合，此时需要增大风险上限。
        """
    )

    with st.expander("查看 12 个月资产收益率数据"):
        st.dataframe(get_return_dataframe(), use_container_width=True)

    st.subheader("灵敏度分析")
    if st.button("计算不同 D 值下的最优年化收益率"):
        sensitivity_limits = [0.05, 0.10, 0.15, 0.20, 0.30, 0.60, 0.90, 1.20]
        sensitivity_df = run_sensitivity_analysis(sensitivity_limits)
        st.dataframe(sensitivity_df, use_container_width=True, hide_index=True)

        feasible_df = sensitivity_df.dropna(subset=["最优年化收益率（%）"])
        if feasible_df.empty:
            st.warning("这些 D 值下均无可行解，请尝试更高的风险上限。")
        else:
            sensitivity_chart = px.line(
                feasible_df,
                x="MAD风险上限（%）",
                y="最优年化收益率（%）",
                markers=True,
            )
            sensitivity_chart.update_layout(
                xaxis_title="风险上限 D（%）",
                yaxis_title="最优年化收益率（%）",
            )
            st.plotly_chart(sensitivity_chart, use_container_width=True)

    st.divider()
    st.caption("本工具仅用于运筹学课程设计与模型演示，不构成任何真实投资建议。")


if __name__ == "__main__":
    main()
