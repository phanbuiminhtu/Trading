import textwrap
import pandas as pd
import google.generativeai as genai
from vnstock import Finance
from IPython.display import display, Markdown
from getData import fetch_from_vnstock
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Thư viện mới để vẽ biểu đồ và xử lý ảnh
import matplotlib.pyplot as plt
import io
from PIL import Image

# --- 1. SETUP AND CONFIGURATION ---

def configure_api():
    """
    Configures the Google Generative AI API.
    """
    try:
        # Thay thế "YOUR_API_KEY" bằng khóa API thực của bạn
        api_key = "YOUR_API_KEY"
        if not api_key or api_key == "YOUR_API_KEY":
            print("ERROR: GOOGLE_API_KEY is not set or is a placeholder.")
            print("Please set your API key to proceed.")
            return None
        genai.configure(api_key=api_key)
        # Sử dụng model Pro để có khả năng phân tích hình ảnh tốt hơn
        return genai.GenerativeModel('gemini-2.5-flash')
    except Exception as e:
        print(f"An error occurred during API configuration: {e}")
        return None

def to_markdown(text):
  """Formats text for nice display in notebooks."""
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

# --- 2. AGENT DEFINITIONS (AS FUNCTIONS) ---

def agent_gather_data(symbol: str) -> dict:
    """
    Agent 1: Data Gatherer (Fundamental).
    """
    print(f"📈 [Data Agent] Đang lấy dữ liệu tài chính cho {symbol}...")
    try:
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)

        income_df = Finance(symbol=symbol, source='VCI').income_statement(period='quarter', lang='vi')
        balance_df = Finance(symbol=symbol, source='VCI').balance_sheet(period='quarter', lang='vi')
        cash_flow_df = Finance(symbol=symbol, source='VCI').cash_flow(period='quarter', lang='vi')
        ratio_df = Finance(symbol=symbol, source='VCI').ratio(period='quarter', lang='vi')

        print(f"✅ [Data Agent] Đã lấy dữ liệu cơ bản thành công cho {symbol}.")
        return {
            "income_statement": income_df,
            "balance_sheet": balance_df,
            "cash_flow": cash_flow_df,
            "ratios": ratio_df
        }
    except Exception as e:
        error_message = f"❌ [Data Agent] Lỗi khi lấy dữ liệu cơ bản cho {symbol}: {e}"
        print(error_message)
        return {}

# =================================================================
# >>> AGENT ĐÃ ĐƯỢC NÂNG CẤP HOÀN TOÀN <<<
# =================================================================
def agent_technical_analysis(model, symbol: str) -> str:
    """
    Agent 1.5: Technical Analyst (Image-based).
    Reads historical price data, generates a chart, and sends the image for analysis.
    """
    print(f"📉 [Technical Agent] Đang tạo và phân tích biểu đồ kỹ thuật cho {symbol}...")
    end = datetime.now()
    start = end - relativedelta(years=5)
    fetch_from_vnstock(symbol,"1D", start, end)
    file_name = f"price_data/{symbol}_1D.csv"
    try:
        # 1. Đọc và chuẩn bị dữ liệu
        price_df = pd.read_csv(file_name)
        price_df['datetime'] = pd.to_datetime(price_df['datetime'])
        price_df = price_df.sort_values('datetime', ascending=True) # Sắp xếp từ cũ đến mới để vẽ biểu đồ

        # 2. Vẽ biểu đồ
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle(f'Biểu đồ Giá và Khối lượng của {symbol} (5 năm)', fontsize=16)

        # Biểu đồ giá
        ax1.plot(price_df['datetime'], price_df['close'], label='Giá đóng cửa', color='blue')
        ax1.set_ylabel('Giá (VND)')
        ax1.grid(True)
        ax1.legend()

        # Biểu đồ khối lượng
        ax2.bar(price_df['datetime'], price_df['volume'], label='Khối lượng', color='gray', alpha=0.7)
        ax2.set_ylabel('Khối lượng')
        ax2.set_xlabel('Ngày')
        ax2.grid(True)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # 3. Lưu biểu đồ vào bộ nhớ đệm (in-memory buffer)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # 4. Tạo đối tượng hình ảnh để gửi cho API
        img = Image.open(buf)
        plt.show()
        #plt.close(fig) # Đóng biểu đồ để giải phóng bộ nhớ

        # 5. Tạo prompt mới và gửi cho AI cùng với hình ảnh
        prompt = f"""
        Bạn là một Chuyên viên Phân tích Kỹ thuật cao cấp. Dựa vào hình ảnh biểu đồ giá và khối lượng trong 5 năm của cổ phiếu {symbol} được cung cấp, hãy đưa ra một phân tích chi tiết:

        1.  **Xu hướng dài hạn (Multi-year Trend):** Xác định xu hướng chính trong toàn bộ giai đoạn (tăng, giảm, đi ngang).
        2.  **Các chu kỳ chính:** Cổ phiếu đã trải qua những chu kỳ tăng/giảm giá lớn nào?
        3.  **Vùng hỗ trợ/kháng cự dài hạn:** Xác định các vùng giá quan trọng đã đóng vai trò là hỗ trợ hoặc kháng cự mạnh trong quá khứ.
        4.  **Phân tích khối lượng:** Có những giai đoạn nào khối lượng giao dịch tăng đột biến không? Nó tương quan với biến động giá như thế nào? (Ví dụ: khối lượng lớn tại đỉnh/đáy).
        5.  **Kết luận tổng quan:** Dựa trên bức tranh toàn cảnh, vị thế hiện tại của cổ phiếu là gì (đang ở đầu, giữa hay cuối một chu kỳ)? Có tiềm năng hay rủi ro gì lớn trong dài hạn không?
        """
        
        print(f"✅ [Technical Agent] Đã tạo biểu đồ, đang gửi cho AI phân tích...")
        response = model.generate_content([prompt, img])
        
        buf.close() # Đóng buffer
        print(f"✅ [Technical Agent] Đã hoàn thành phân tích kỹ thuật cho {symbol}.")
        return response.text

    except FileNotFoundError:
        error_message = f"⚠️ [Technical Agent] Không tìm thấy file dữ liệu giá: '{file_name}'. Bỏ qua bước phân tích kỹ thuật."
        print(error_message)
        return error_message
    except Exception as e:
        error_message = f"❌ [Technical Agent] Lỗi khi phân tích kỹ thuật cho {symbol}: {e}"
        print(error_message)
        return error_message


def agent_analyze_financials(model, financial_data_dict: dict, technical_analysis_report: str, symbol: str) -> str:
    """
    Agent 2: Financial Analyst (Synthesizer).
    """
    print(f"📑 [Analyst Agent] Bắt đầu phân tích tuần tự cho {symbol}...")
    all_analyses = []

    # --- Phân tích các báo cáo tài chính (đã rút gọn cho dễ đọc) ---
    print("   [1/4] Phân tích Báo cáo kết quả kinh doanh...")
    income_df_str = financial_data_dict['income_statement'].to_string()
    prompt1 = f"""Bạn là chuyên gia phân tích tài chính chuyên về doanh nghiệp bất động sản Việt Nam.  
        Tôi sẽ cung cấp dữ liệu Báo cáo kết quả kinh doanh (Income Statement) của {symbol}, một doanh nghiệp BĐS:

        {income_df_str}

        Hãy phân tích theo hướng:
        1. **Hiệu quả hoạt động kinh doanh**: xu hướng doanh thu, lợi nhuận gộp, lợi nhuận thuần qua các năm.  
        2. **Cấu trúc lợi nhuận**: mức độ phụ thuộc vào thu nhập tài chính, chi phí lãi vay, lợi nhuận khác.  
        3. **Biên lợi nhuận**:  
        - Biên lợi nhuận gộp = Lãi gộp / Doanh thu thuần.  
        - Biên lợi nhuận ròng = Lợi nhuận sau thuế / Doanh thu thuần.  
        4. **Tăng trưởng và ổn định lợi nhuận**: tốc độ tăng trưởng doanh thu & lợi nhuận; biến động có ổn định không?  
        5. **Nhận xét theo đặc thù doanh nghiệp BĐS**:  
        - Lợi nhuận có đến từ bàn giao dự án hay chủ yếu từ tài chính?  
        - Chu kỳ lợi nhuận có bị gián đoạn theo dự án không?  
        6. **Kết luận ngắn gọn**:  
        - Hiệu quả kinh doanh: mạnh / trung bình / yếu.  
        - Lợi nhuận có bền vững không?  

        Đầu ra mong muốn:
        - Bảng tóm tắt theo từng năm.  
        - Diễn giải xu hướng rõ ràng, ngắn gọn, có logic.  
        - Phong cách báo cáo phân tích đầu tư chuyên nghiệp.
    """

    all_analyses.append(f"### 1. Phân tích Kết quả Kinh doanh\n{model.generate_content(prompt1).text}")

    print("   [2/4] Phân tích Bảng cân đối kế toán...")
    balance_df_str = financial_data_dict['balance_sheet'].to_string()
    prompt2 = f"""Bạn là chuyên gia phân tích tài chính chuyên về doanh nghiệp bất động sản Việt Nam.  
        Tôi sẽ cung cấp dữ liệu **Bảng cân đối kế toán (Balance Sheet)** của doanh nghiệp {symbol}:

        {balance_df_str}

        Hãy phân tích tập trung vào:
        1. **Cơ cấu tài sản và nợ**: tỷ trọng tài sản ngắn hạn / dài hạn; nợ phải trả / vốn chủ sở hữu.  
        2. **Phân tích nợ vay**:  
        - So sánh nợ ngắn hạn vs nợ dài hạn.  
        - Mức độ phụ thuộc vào vay ngắn hạn.  
        - Rủi ro thanh khoản và tái cấp vốn.  
        3. **Đánh giá hàng tồn kho và dự án**:  
        - Tỷ trọng hàng tồn kho trong tổng tài sản.  
        - Tồn kho tăng do mở rộng dự án hay do chậm tiêu thụ?  
        - “Người mua trả tiền trước” → phản ánh tiến độ bán dự án.  
        4. **Khả năng thanh toán và dòng tiền**:  
        - Hệ số thanh toán hiện hành = Tài sản ngắn hạn / Nợ ngắn hạn.  
        - Hệ số thanh toán nhanh = (Tài sản ngắn hạn – Hàng tồn kho) / Nợ ngắn hạn.  
        5. **Đòn bẩy tài chính**: xu hướng tỷ lệ nợ/vốn chủ, rủi ro lãi suất.  
        6. **Tổng hợp đánh giá**:  
        - Cấu trúc tài chính an toàn, cân bằng hay rủi ro cao?  
        - Doanh nghiệp đang mở rộng, ổn định hay thu hẹp quy mô đầu tư?

        Đầu ra mong muốn:
        - Bảng hoặc đoạn tóm tắt theo từng năm.  
        - Biểu đồ hoặc mô tả xu hướng (nếu có thể).  
        - Giọng văn khách quan, phong cách phân tích đầu tư chuyên nghiệp.
    """

    all_analyses.append(f"### 2. Phân tích Bảng cân đối kế toán\n{model.generate_content(prompt2).text}")

    print("   [3/4] Phân tích Báo cáo lưu chuyển tiền tệ...")
    cash_flow_df_str = financial_data_dict['cash_flow'].to_string()
    prompt3 = f"""Bạn là chuyên gia phân tích tài chính chuyên về doanh nghiệp bất động sản Việt Nam.  
        Tôi sẽ cung cấp cho bạn dữ liệu **Báo cáo lưu chuyển tiền tệ (Cash Flow Statement)** của {symbol}, một doanh nghiệp BĐS:

        {cash_flow_df_str}

        Hãy phân tích theo hướng:
        1. **Dòng tiền hoạt động kinh doanh (Operating Cash Flow – OCF)**  
        - So sánh xu hướng OCF qua các năm: dương hay âm?  
        - Nếu OCF âm kéo dài, nguyên nhân là gì (tăng tồn kho, phải thu, hay lợi nhuận không chuyển thành tiền)?  
        - Đối với doanh nghiệp BĐS, lưu ý: OCF có thể âm trong giai đoạn đầu tư dự án – hãy đánh giá tính chu kỳ này.

        2. **Dòng tiền đầu tư (Investing Cash Flow – ICF)**  
        - Phân tích chi cho mua TSCĐ, đầu tư dự án, hoặc đầu tư tài chính.  
        - Có dấu hiệu **mở rộng đầu tư** (chi ra nhiều) hay **thu hẹp/quay vòng vốn** (thu hồi đầu tư, thanh lý tài sản)?  
        - Nhận xét về tính hợp lý giữa dòng tiền đầu tư và chiến lược phát triển doanh nghiệp.

        3. **Dòng tiền tài chính (Financing Cash Flow – FCF)**  
        - Phân tích nguồn tiền đến từ vay nợ, phát hành cổ phiếu, và chi ra cho trả nợ, trả cổ tức.  
        - Doanh nghiệp có phụ thuộc nhiều vào **dòng tiền vay nợ** không?  
        - Dòng tiền tài chính dương do vay mới hay do huy động vốn cổ phần?

        4. **Tổng hợp và đánh giá dòng tiền thuần (Net Cash Flow)**  
        - Tiền và tương đương tiền cuối kỳ có xu hướng tăng hay giảm?  
        - Dòng tiền có phản ánh đúng lợi nhuận kế toán không (lợi nhuận cao nhưng tiền âm)?  
        - Đánh giá **khả năng trả nợ và duy trì thanh khoản** trong bối cảnh thị trường BĐS chậm.

        5. **Kết luận theo đặc thù BĐS**  
        - Doanh nghiệp đang ở giai đoạn: mở rộng dự án / thu hồi vốn / tái cơ cấu nợ.  
        - Dòng tiền có lành mạnh không?  
        - Rủi ro tiềm ẩn về thanh khoản hoặc đòn bẩy vốn ngắn hạn?

        Đầu ra mong muốn:
        - Bảng hoặc đoạn tóm tắt cho từng năm.  
        - Phân tích rõ nguyên nhân chính ảnh hưởng đến từng loại dòng tiền.  
        - Biểu đồ xu hướng 3 dòng tiền (hoạt động – đầu tư – tài chính) nếu có thể.  
        - Phong cách báo cáo phân tích đầu tư chuyên nghiệp, khách quan.
    """

    all_analyses.append(f"### 3. Phân tích Lưu chuyển tiền tệ\n{model.generate_content(prompt3).text}")

    # --- Final Step: Synthesis and Conclusion ---
    print("   [4/4] Tổng hợp Phân tích Cơ bản và Kỹ thuật...")
    try:
        ratios_df_str = financial_data_dict['ratios'].to_string()
        previous_analyses = "\n\n".join(all_analyses)
        prompt4 = f"""
        Bạn là Chuyên viên Phân tích Đầu tư cao cấp, kết hợp cả phân tích cơ bản và kỹ thuật.
        Nhiệm vụ của bạn là đưa ra một kết luận cuối cùng cho nhà đầu tư ngắn hạn (1-3 tháng) về cổ phiếu {symbol}.

        **Phần 1: Các phân tích chi tiết về tài chính doanh nghiệp (Phân tích cơ bản):**
        {previous_analyses}

        **Phần 2: Phân tích biểu đồ dài hạn (Phân tích kỹ thuật):**
        {technical_analysis_report}

        **Phần 3: Dữ liệu về các chỉ số tài chính quan trọng:**
        {ratios_df_str}

        ---
        **HƯỚNG DẪN TỔNG HỢP CUỐI CÙNG:**
        Dựa trên TẤT CẢ thông tin trên, hãy viết một báo cáo tổng hợp có cấu trúc:

        **1. Luận điểm đầu tư tổng hợp:**
           - Kết hợp cả hai góc nhìn, tóm tắt câu chuyện đầu tư chính của {symbol} hiện tại là gì?

        **2. Đánh giá theo góc nhìn cơ bản (Fundamental):**
           - **Cơ hội:** Điểm sáng nhất về tài chính là gì?
           - **Rủi ro:** Rủi ro lớn nhất về tài chính là gì?

        **3. Đánh giá theo góc nhìn kỹ thuật (Technical):**
            - **Tín hiệu Tích cực:** Xu hướng dài hạn, vùng hỗ trợ mạnh,...
            - **Tín hiệu Tiêu cực:** Vùng kháng cự mạnh, mẫu hình giá xấu,...

        **4. Kết luận và Khuyến nghị (1-3 tháng):**
           - **Kết hợp tất cả các yếu tố**, đưa ra đánh giá cuối cùng: Cổ phiếu này đang **HẤP DẪN**, **TRUNG LẬP**, hay **KHÔNG HẤP DẪN**.
           - **Giải thích rõ ràng** lý do cho khuyến nghị của bạn.
        """
        final_response = model.generate_content(prompt4)
        print(f"✅ [Analyst Agent] Đã hoàn thành phân tích tổng hợp cho {symbol}.")
        return final_response.text
    except Exception as e:
        error_message = f"❌ [Analyst Agent] Lỗi trong quá trình tổng hợp phân tích: {e}"
        print(error_message)
        return error_message

def agent_generate_investment_summary(model, analysis_report: str, symbol: str) -> str:
    # (Hàm này không thay đổi)
    print(f"✍️ [Advisor Agent] Đang tạo tóm tắt đầu tư cho {symbol}...")
    prompt = f"""
    Bạn là một Cố vấn Đầu tư. Hãy tổng hợp báo cáo phân tích chi tiết sau đây cho công ty {symbol} thành một bản tóm tắt đầu tư rõ ràng, có thể hành động.
    Báo cáo Phân tích Chi tiết: {analysis_report}
    ---
    **NHIỆM VỤ:**
    Viết một bản tóm tắt đầu tư cuối cùng theo cấu trúc:
    **1. Tóm tắt (2-3 câu):** Mô tả ngắn gọn sức khỏe tài chính và tình hình giá cổ phiếu.
    **2. Điểm mạnh chính (Luận điểm Tăng giá):** Liệt kê 2-3 điểm tích cực nhất.
    **3. Điểm yếu/Rủi ro chính (Luận điểm Giảm giá):** Liệt kê 2-3 rủi ro lớn nhất.
    **4. Kết luận & Khuyến nghị:** Đưa ra luận điểm đầu tư rõ ràng và lý do.
    """
    try:
        response = model.generate_content(prompt)
        print(f"✅ [Advisor Agent] Đã hoàn thành tóm tắt cho {symbol}.")
        return response.text
    except Exception as e:
        return f"❌ [Advisor Agent] Lỗi trong quá trình tạo tóm tắt: {e}"


# --- 3. MAIN ORCHESTRATION ---

def main(symbol: str):
    # (Hàm này không thay đổi)
    stock_symbol_to_analyze = symbol
    print("--- Bắt đầu Phân tích Cổ phiếu Toàn diện ---")
    model = configure_api()
    if model is None: return

    financial_data_dictionary = agent_gather_data(stock_symbol_to_analyze)
    if not financial_data_dictionary: return

    technical_report = agent_technical_analysis(model, stock_symbol_to_analyze)
    if "Lỗi" in technical_report and "Không tìm thấy file" not in technical_report: return

    detailed_analysis = agent_analyze_financials(model, financial_data_dictionary, technical_report, stock_symbol_to_analyze)
    if "Lỗi" in detailed_analysis: return

    investment_summary = agent_generate_investment_summary(model, detailed_analysis, stock_symbol_to_analyze)
    if "Lỗi" in investment_summary: return

    # --- 4. DISPLAY FINAL RESULTS ---
    print("\n\n==================================================")
    print("          BÁO CÁO PHÂN TÍCH TOÀN DIỆN")
    print("==================================================\n")
    display(detailed_analysis)

    print("\n\n==================================================")
    print("         TÓM TẮT TỪ CỐ VẤN ĐẦU TƯ")
    print("==================================================\n")
    display(investment_summary)


if __name__ == '__main__':
    # <<< THAY ĐỔI MÃ CỔ PHIẾU BẠN MUỐN PHÂN TÍCH TẠI ĐÂY >>>
    # Đảm bảo bạn có file "FPT_1D.csv" trong cùng thư mục

    main("VRE")
