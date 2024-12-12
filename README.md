# Documentation for the Development of a Financial Chatbot with LLM Models

This project aims to build a chatbot based on LLM models in Spanish to answer queries related to financial products for a banking entity. The chatbot should process documentation in diverse PDF formats, convert it into structured data, and provide accurate responses to questions posed by employees and customers.

## **Project Scope**

- **Chatbot Features:**
  - Ability to answer questions about:
    - **International Transfers**
    - **Investment Funds**
    - **Active Loans**
    - **Investment Options**

- **Data Sources:**
  - Provided PDF documents in diverse formats, organized in the `raw/` folder:
    - `W1_Tarifas transferencias Extranjero.pdf`
    - `W2_Ficha Tecnica.pdf`
    - `W3_Catalogo de productos de activo vigentes.pdf`
    - `W4_2023_12_Posicionamiento_Environment-1.pdf`
    - `W5_2023_12_Posicionamiento_Environment-2.pdf`

- **Data Processing:**
  - Conversion of PDF documents into `.docx` format to facilitate extraction.
  - Transformation into JSON structures, with data categorized into:
    - Text Sections: titles and content.
    - Structured Tables: titles and organized data.

## **Example Questions Answered**

1. **Transfers outside the Euro Zone:**
   - Can you calculate the fees for a transfer outside the Euro Zone of €10,000 with the fees borne by the partner?

2. **Investment Funds:**
   - What is the risk level of the investment fund CI Environment ISR?

3. **Active Loans:**
   - What are the requirements to apply for a postgraduate loan?

---

This chatbot will be a valuable tool for employees and customers, simplifying access to financial information and enhancing user experience.
