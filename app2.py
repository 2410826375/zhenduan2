import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from zhenduanzip import AcousticDiagnoser, DIAG_CONFIG

# 初始化诊断器
diagnoser = AcousticDiagnoser(DIAG_CONFIG)

st.set_page_config(page_title="基于声纹信号的变压器故障诊断系统", layout="centered")
st.title("基于声纹信号的变压器故障诊断系统")
st.markdown("请上传 .npy 格式的 MEL 时频图文件或 .xlsx 格式的原始数据文件：")

uploaded_file = st.file_uploader("请选择一个文件", type=["npy", "xlsx"])
if uploaded_file is not None:
    temp_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ 文件已上传成功：{uploaded_file.name}")

    # 🎵 显示原始波形图或MEL图
    st.subheader("原始信号波形图")
    try:
        if uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(temp_path)
            time = df["Time"].values
            amplitude = df["Amplitude"].values
            fig, ax = plt.subplots()
            ax.plot(time, amplitude, linewidth=0.8)
            ax.set_title("声纹信号波形图")
            ax.set_xlabel("时间（秒）")
            ax.set_ylabel("幅值")
            st.pyplot(fig)

        #elif uploaded_file.name.endswith(".npy"):
        #    mel = np.load(temp_path)
        #    fig, ax = plt.subplots()
        #    ax.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
         #   ax.set_title("MEL 时频图")
         #   ax.set_xlabel("帧")
         #   ax.set_ylabel("Mel 滤波器通道")
         #   st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ 波形图显示失败：{str(e)}")

    # 诊断按钮
    if st.button("开始诊断"):
        with st.spinner("模型诊断中，请稍候..."):
            try:
                result = diagnoser.diagnose(temp_path)
            except Exception as e:
                st.error(f"❌ 诊断失败：{str(e)}")
                result = None

        if result is not None:
            if isinstance(result, list):  # 多段诊断（如 Excel 分段）
                st.subheader("🧠 多段诊断结果")

                consecutive_fault = None
                count = 0
                overall_diagnosis = None

                for i, segment_result in enumerate(result):
                    st.markdown(f"### 📌 第 {i + 1} 秒")
                    st.write(f"**预测故障类型：** {segment_result['预测故障类型']}")
                    st.write(f"**置信度：** {segment_result['置信度']:.4f}")

                    if consecutive_fault == segment_result["预测故障类型"]:
                        count += 1
                    else:
                        consecutive_fault = segment_result["预测故障类型"]
                        count = 1

                    if count >= 3 and overall_diagnosis is None:
                        overall_diagnosis = consecutive_fault
                        st.subheader("📌 整个文件诊断结果")
                        st.write(f"**预测故障类型：** {overall_diagnosis}")

                    img_name = f"diagnosis_{os.path.splitext(os.path.basename(temp_path))[0]}_segment{i + 1}.png"
                    img_path = os.path.join(DIAG_CONFIG["output_dir"], img_name)
                    if os.path.exists(img_path):
                        st.image(Image.open(img_path), caption=f"可视化图 - 第 {i + 1} 秒", use_column_width=True)

                if overall_diagnosis is None:
                    st.subheader("📌 整个文件诊断结果")
                    st.write(f"**预测故障类型：** {result[-1]['预测故障类型']}")

                st.subheader("📋 诊断汇总报告（表格）")
                st.dataframe(result)

            elif isinstance(result, dict):  # 单段诊断（npy）
                st.subheader("🧠 诊断结果")
                st.write(f"**预测故障类型：** {result['预测故障类型']}")
                st.write(f"**置信度：** {result['置信度']:.4f}")

                img_name = f"diagnosis_{os.path.splitext(os.path.basename(temp_path))[0]}_segmentnpy.png"
                img_path = os.path.join(DIAG_CONFIG["output_dir"], img_name)
                if os.path.exists(img_path):
                    st.subheader("📊 概率分布图")
                    st.image(Image.open(img_path), caption="诊断可视化图", use_column_width=True)

                st.subheader("📋 诊断报告（表格）")
                st.dataframe(result)
            else:
                st.error("❌ 未知诊断结果格式")
        else:
            st.error("❌ 未能获取诊断结果")
