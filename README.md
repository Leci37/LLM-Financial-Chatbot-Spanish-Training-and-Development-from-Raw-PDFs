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
   - ¿Puedes calcularme las comisiones para una transferencia fuera de la Zona €, de 10.000€ con las comisiones a cargo del socio?
   _- Can you calculate the fees for a transfer outside the Euro Zone of €10,000 with the fees borne by the partner?_

3. **Investment Funds:**
   - ¿Qué nivel de riesgo tiene el fondo de inversión, CI Environment ISR?
   _- What is the risk level of the investment fund CI Environment ISR?_

5. **Active Loans:**
   - ¿Qué requisitos hay que cumplir para solicitar un préstamo postgrado?
   _- What are the requirements to apply for a postgraduate loan?_

This chatbot will be a valuable tool for employees and customers, simplifying access to financial information and enhancing user experience.
