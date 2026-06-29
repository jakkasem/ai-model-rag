# =============================================================================
# Project  : DOS RAG - PDF Policy Importer
# File     : readPDFInsertData_thaiOCR_preprocess.py
# Author   : <Your Name>
# Purpose  :
#   โปรแกรมสำหรับนำเอกสาร PDF เข้า PostgreSQL + pgvector
#   เพื่อใช้เป็น Knowledge Base สำหรับระบบ RAG (Retrieval Augmented Generation)
#
# =============================================================================
# Workflow การทำงาน
# =============================================================================
#
#   PDF File
#      │
#      ├─ กรณี PDF มี Text Layer
#      │      ↓
#      │   pdfplumber
#      │      ↓
#      │   Extract Text + Table
#      │
#      └─ กรณี Scan PDF
#             ↓
#       pdf2image
#             ↓
#       Convert PDF → Image
#             ↓
#       Image Preprocessing
#             ├─ Grayscale
#             ├─ Denoise
#             ├─ Adaptive Threshold
#             └─ Resize x1.5
#             ↓
#       Tesseract OCR
#             ↓
#       Thai + English Text
#             ↓
#       Chunking
#             ↓
#       E5 Embedding
#             ↓
#       PostgreSQL pgvector
#
# =============================================================================
# เหตุผลที่ใช้ Hybrid Extraction
# =============================================================================
#
# PDF มีอยู่ 2 ประเภท
#
# 1. Text PDF
#    - สร้างจาก Word / Excel / PowerPoint
#    - มีข้อความอยู่ภายใน PDF จริง
#    - pdfplumber สามารถอ่านได้โดยตรง
#    - แม่นยำกว่า OCR
#
# 2. Scan PDF
#    - เกิดจากการ Scan เอกสารกระดาษ
#    - ภายใน PDF เป็นรูปภาพ
#    - pdfplumber อ่านข้อความไม่ได้
#    - ต้องใช้ OCR แปลงรูปเป็นข้อความ
#
# โปรแกรมจะพยายามอ่านด้วย pdfplumber ก่อนเสมอ
#
# ถ้าอ่านได้มากกว่า 300 ตัวอักษร
#     → ใช้ข้อมูลจาก pdfplumber
#
# ถ้าอ่านไม่ได้
#     → เปลี่ยนไปใช้ OCR อัตโนมัติ
#
# =============================================================================
# OCR Preprocessing
# =============================================================================
#
# ก่อนส่งภาพเข้า OCR จะมีการปรับคุณภาพภาพก่อน
#
# 1. Grayscale
#    แปลงภาพสีเป็นขาวดำ
#
# 2. Denoise
#    ลด Noise จากการ Scan
#
# 3. Adaptive Threshold
#    ทำตัวอักษรให้คมขึ้น
#
# 4. Resize 150%
#    ขยายตัวอักษรให้ OCR อ่านง่ายขึ้น
#
# ช่วยเพิ่มความแม่นยำสำหรับเอกสารภาษาไทย
# โดยเฉพาะเอกสาร HR Policy และเอกสาร Scan เก่า
#
# =============================================================================
# OCR Configuration
# =============================================================================
#
# OCR Engine : Tesseract OCR
#
# Language :
#     tha + eng
#
# รองรับ
#     ภาษาไทย
#     ภาษาอังกฤษ
#     เอกสารผสม Thai / English
#
# Tesseract Parameters
#
# --oem 3
#     ใช้ OCR Engine ที่ดีที่สุดที่มี
#
# --psm 6
#     มองเอกสารเป็น Block ของข้อความ
#
# preserve_interword_spaces=1
#     พยายามรักษาระยะห่างระหว่างคำ
#
# =============================================================================
# Chunking Strategy
# =============================================================================
#
# หลัง OCR จะได้ข้อความขนาดใหญ่
#
# ต้องแบ่งเป็น Chunk ก่อนสร้าง Embedding
#
# Config
#
# chunk_size    = 1200
# chunk_overlap = 250
#
# เหตุผล
#
# - ลดการสูญเสีย Context
# - เพิ่มคุณภาพในการค้นหา
# - เหมาะกับ Policy / Procedure Document
#
# =============================================================================
# Embedding Model
# =============================================================================
#
# Model:
#     intfloat/multilingual-e5-base
#
# ข้อดี
#
# - รองรับภาษาไทย
# - รองรับภาษาอังกฤษ
# - เหมาะกับ RAG
# - Retrieval Accuracy สูง
#
# การสร้าง Embedding
#
#     passage: <text>
#
# ตามคำแนะนำของ E5 Model
#
# =============================================================================
# Database Storage
# =============================================================================
#
# ตาราง:
#     public.fujitsu_hr_policy
#
# Columns
#
#     title
#         ชื่อไฟล์ PDF
#
#     content
#         ข้อความของแต่ละ Chunk
#
#     embedding
#         Vector Embedding สำหรับค้นหา
#
# =============================================================================
# External Dependencies
# =============================================================================
#
# Python Packages
#
#     pdfplumber
#     pdf2image
#     pytesseract
#     pillow
#     opencv-python
#     numpy
#     sentence-transformers
#     langchain-text-splitters
#     psycopg2-binary
#
# External Software
#
#     Tesseract OCR
#         C:\Program Files\Tesseract-OCR
#
#     Poppler
#         C:\poppler\Library\bin
#
# ตรวจสอบการติดตั้ง
#
#     tesseract --version
#     tesseract --list-langs
#     pdfinfo -v
#
# =============================================================================
# Expected Result
# =============================================================================
#
# Input
#
#     FY2016_22.-Leaves-Policy.pdf
#
# Output
#
#     PDF
#       ↓
#     Text
#       ↓
#     Chunk
#       ↓
#     Embedding
#       ↓
#     PostgreSQL pgvector
#
# พร้อมใช้งานในระบบ RAG / Semantic Search / Chatbot
#
# =============================================================================

import os
import psycopg2
import pdfplumber
import pytesseract
import cv2
import numpy as np

from pdf2image import convert_from_path
from PIL import Image
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------
# Tesseract
# --------------------------------------------------
TESSERACT_PATH = os.getenv("TESSERACT_PATH", "/usr/bin/tesseract")
POPPLER_PATH = os.getenv("POPPLER_PATH", None)

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

MODEL = SentenceTransformer("intfloat/multilingual-e5-large")


def preprocess_image(pil_image):
    """
    OCR Preprocessing
    - grayscale
    - denoise
    - adaptive threshold
    - enlarge
    """

    img = np.array(pil_image)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    denoise = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=15
    )

    thresh = cv2.adaptiveThreshold(
        denoise,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15
    )

    scale = 1.5

    resized = cv2.resize(
        thresh,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    return Image.fromarray(resized)


def extract_pdf_with_ocr(file_path, dpi=400):

    print("Using OCR with preprocessing (tha+eng)...")

    pages = convert_from_path(
        file_path,
        dpi=dpi,
        poppler_path=POPPLER_PATH,
    )

    result = []

    custom_config = (
        "--oem 3 "
        "--psm 6 "
        "-c preserve_interword_spaces=1"
    )

    for page_no, page in enumerate(pages, start=1):

        processed = preprocess_image(page)

        text = pytesseract.image_to_string(
            processed,
            lang="tha+eng",
            config=custom_config
        )

        result.append(
            f"--- Page {page_no} ---\n{text}"
        )

    return "\n\n".join(result)


def extract_pdf_content(file_path):

    full_content = []

    try:

        print(
            f"Trying pdfplumber extraction: "
            f"{os.path.basename(file_path)}"
        )

        with pdfplumber.open(file_path) as pdf:

            for page_num, page in enumerate(pdf.pages):

                page_parts = []

                text = page.extract_text()

                if text:
                    page_parts.append(text)

                tables = page.extract_tables()

                if tables:

                    page_parts.append("\n[Table Data]\n")

                    for table in tables:

                        if not table:
                            continue

                        markdown = ""

                        for i, row in enumerate(table):

                            clean_row = [
                                str(cell).replace("\n", " ").strip()
                                if cell else ""
                                for cell in row
                            ]

                            markdown += (
                                "| "
                                + " | ".join(clean_row)
                                + " |\n"
                            )

                            if i == 0:

                                markdown += (
                                    "| "
                                    + " | ".join(
                                        ["---"] * len(clean_row)
                                    )
                                    + " |\n"
                                )

                        page_parts.append(markdown)

                if page_parts:

                    full_content.append(
                        f"--- Page {page_num+1} ---\n"
                        + "\n".join(page_parts)
                    )

        extracted = "\n\n".join(full_content)

        if len(extracted.strip()) > 300:

            print(
                f"pdfplumber extracted "
                f"{len(extracted)} chars"
            )

            return extracted

    except Exception as e:

        print(f"pdfplumber failed: {e}")

    return extract_pdf_with_ocr(file_path)


def chunk_text(
    content,
    # chunk_size=1200,
    # chunk_overlap=250
    chunk_size=800,
    chunk_overlap=150
):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    return splitter.split_text(content)


def process_pdf_to_db(filepath):

    db_url = os.getenv("DB_URL", "postgresql://postgres:postgres@localhost:5432/mydb")

    conn = None

    try:

        title = os.path.basename(filepath)

        content = extract_pdf_content(filepath)

        chunks = chunk_text(content)

        print(f"Processing {len(chunks)} chunks")

        conn = psycopg2.connect(db_url)
        cur = conn.cursor()

        # ลบข้อมูลเก่าของ title นี้ก่อน 1 ครั้ง (ต้องอยู่นอก loop)
        cur.execute(
            """
            DELETE FROM public.fujitsu_hr_policy
            WHERE title = %s
            """,
            (title,)
        )
        print(f"Deleted old data for: {title}")

        inserted = 0

        total_chunks = len([c for c in chunks if len(c.strip()) >= 20])
        print(f"Valid chunks to insert: {total_chunks}")

        for chunk in chunks:

            if len(chunk.strip()) < 20:
                continue

            embedding = MODEL.encode(
                f"passage: {chunk}"
            ).tolist()

            cur.execute(
                """
                INSERT INTO public.fujitsu_hr_policy
                (
                    title,
                    content,
                    embedding
                )
                VALUES
                (
                    %s,
                    %s,
                    %s::public.vector
                )
                """,
                (
                    title,
                    chunk,
                    embedding
                )
            )

            inserted += 1

        conn.commit()

        print(f"Successfully inserted {inserted} chunks for: {title}")

    except Exception as e:

        print(f"Error occurred: {e}")

        if conn:
            conn.rollback()

    finally:

        if conn:
            cur.close()
            conn.close()


def process_folder(folder_path: str):
    """index PDF ทุกไฟล์ใน folder"""
    pdf_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".pdf")
    ]
    print(f"Found {len(pdf_files)} PDF files in {folder_path}")
    for i, fp in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Processing: {os.path.basename(fp)}")
        process_pdf_to_db(fp)
    print("\nAll files processed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Index PDF files into PostgreSQL pgvector")
    parser.add_argument("--file",   type=str, help="index ไฟล์เดียว")
    parser.add_argument("--folder", type=str, help="index ทุกไฟล์ใน folder")
    args = parser.parse_args()

    if args.file:
        if os.path.exists(args.file):
            process_pdf_to_db(args.file)
        else:
            print(f"File not found: {args.file}")
    elif args.folder:
        if os.path.isdir(args.folder):
            process_folder(args.folder)
        else:
            print(f"Folder not found: {args.folder}")
    else:
        filepath = os.getenv("PDF_FILE", "AnnualReport_2568.pdf")
        if os.path.exists(filepath):
            process_pdf_to_db(filepath)
        else:
            print(f"File not found: {filepath}")
