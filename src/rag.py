import os
from pathlib import Path
from typing import Any, Dict, List

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).parent.parent / "data"
PDF_PATH = DATA_DIR / "catalog.pdf"
VECTORSTORE_PATH = DATA_DIR / "vectorstore"

# ---------------------------------------------------------------------------
# Content used to generate the sample PDF catalog
# ---------------------------------------------------------------------------
CATALOG_SECTIONS: Dict[str, List[Dict[str, str]]] = {
    "Engine Parts": [
        {
            "number": "OF-001",
            "name": "Oil Filter Standard",
            "desc": (
                "Standard oil filter for 4-cylinder gasoline engines. "
                "Removes contaminants from engine oil to protect engine "
                "components. Compatible with Toyota, Honda, and Hyundai. "
                "Recommended replacement: every 5,000 miles."
            ),
            "models": "Toyota Camry 2018-2023, Honda Civic 2016-2022, Hyundai Elantra 2017-2023",
            "price": "$12.99",
            "availability": "In Stock",
        },
        {
            "number": "OF-002",
            "name": "Oil Filter Premium Extended Life",
            "desc": (
                "High-performance synthetic media oil filter with extended "
                "life up to 10,000 miles. Features anti-drain-back valve and "
                "silicone gasket for leak-free sealing."
            ),
            "models": "Ford F-150 2015-2023, Chevrolet Silverado 2014-2022, Ram 1500 2013-2022",
            "price": "$19.99",
            "availability": "In Stock",
        },
        {
            "number": "SP-001",
            "name": "Spark Plugs Iridium Set of 4",
            "desc": (
                "Fine-wire iridium tip spark plugs for maximum ignitability "
                "and long service life. Reduces misfire for improved fuel "
                "efficiency and lower emissions."
            ),
            "models": "Honda Civic 2016-2022, Toyota Corolla 2019-2023, Mazda 3 2019-2023",
            "price": "$34.99",
            "availability": "In Stock",
        },
        {
            "number": "TB-001",
            "name": "Timing Belt Kit Complete",
            "desc": (
                "Complete timing belt kit including timing belt, tensioner, "
                "idler pulley, and water pump. Ensures proper engine timing. "
                "Recommended replacement: every 60,000-90,000 miles."
            ),
            "models": "Honda CR-V 2015-2021, Subaru Forester 2014-2018, Acura TLX 2015-2020",
            "price": "$89.99",
            "availability": "Low Stock",
        },
        {
            "number": "SB-001",
            "name": "Serpentine Belt EPDM",
            "desc": (
                "Heavy-duty EPDM serpentine belt driving the alternator, "
                "power steering pump, A/C compressor, and water pump. "
                "Resists heat and cracking."
            ),
            "models": "Ford Mustang 2015-2023, Dodge Charger 2011-2023, Dodge Challenger 2009-2023",
            "price": "$29.99",
            "availability": "In Stock",
        },
        {
            "number": "AF-001",
            "name": "Engine Air Filter High-Flow",
            "desc": (
                "High-flow engine air filter with electrostatic filtration. "
                "Improves airflow to the engine for better performance and "
                "fuel economy. Captures dust, pollen, and airborne particles."
            ),
            "models": "Toyota Camry 2018-2023, Mazda 3 2019-2023, Toyota Corolla 2019-2023",
            "price": "$18.99",
            "availability": "In Stock",
        },
        {
            "number": "THR-001",
            "name": "Electronic Throttle Body 60mm",
            "desc": (
                "Precision electronic throttle body with 60mm bore diameter. "
                "Controls air intake for accurate fuel injection. "
                "Includes gasket and wiring harness connector."
            ),
            "models": "Chevrolet Cruze 2011-2016, Buick Verano 2012-2017, Chevrolet Sonic 2012-2020",
            "price": "$159.99",
            "availability": "Out of Stock",
        },
    ],
    "Brake System": [
        {
            "number": "BP-001",
            "name": "Front Ceramic Brake Pads",
            "desc": (
                "Premium ceramic brake pads for the front axle. Provides "
                "quiet operation, low brake dust, and excellent stopping "
                "power. Includes shims and hardware."
            ),
            "models": "Toyota Camry 2018-2023, Nissan Altima 2019-2023, Mazda 6 2014-2021",
            "price": "$45.99",
            "availability": "In Stock",
        },
        {
            "number": "BP-002",
            "name": "Rear Semi-Metallic Brake Pads",
            "desc": (
                "Semi-metallic rear brake pads offering consistent "
                "performance across a wide temperature range. Enhanced heat "
                "dissipation for repeated stops."
            ),
            "models": "Toyota Camry 2018-2023, Honda Accord 2018-2022, Subaru Legacy 2015-2022",
            "price": "$39.99",
            "availability": "In Stock",
        },
        {
            "number": "BRK-001",
            "name": "Front Vented Brake Rotor",
            "desc": (
                "Vented disc brake rotor with anti-corrosion coating. "
                "Precision machined for smooth, vibration-free braking. "
                "Directional vanes improve cooling for fade resistance."
            ),
            "models": "Toyota Camry 2018-2023, Honda Accord 2018-2022, Nissan Altima 2019-2023",
            "price": "$49.99",
            "availability": "In Stock",
        },
    ],
    "Cooling System": [
        {
            "number": "RAD-001",
            "name": "Aluminum Radiator Complete Assembly",
            "desc": (
                "Full aluminum core radiator with plastic tanks. Direct OE "
                "fit with upper and lower hose connections. Improves cooling "
                "efficiency by 20% over original."
            ),
            "models": "Toyota Camry 2018-2023, Chevrolet Malibu 2016-2022, Toyota RAV4 2019-2023",
            "price": "$189.99",
            "availability": "In Stock",
        },
        {
            "number": "WP-001",
            "name": "Water Pump with Gasket",
            "desc": (
                "Cast iron impeller water pump for reliable coolant "
                "circulation. Includes mounting gasket and hardware. "
                "Ceramic mechanical seal prevents leakage."
            ),
            "models": "BMW 3 Series 2012-2019, BMW 5 Series 2010-2017, BMW X5 2013-2019",
            "price": "$79.99",
            "availability": "In Stock",
        },
        {
            "number": "TH-001",
            "name": "Thermostat with Housing",
            "desc": (
                "Engine thermostat with integrated housing and O-ring seal. "
                "Maintains optimal engine operating temperature of 195 deg F. "
                "Quick-open design prevents overcooling."
            ),
            "models": "Toyota Camry 2018-2023, Toyota Corolla 2019-2023, Lexus ES 2019-2023",
            "price": "$34.99",
            "availability": "In Stock",
        },
    ],
    "Electrical Components": [
        {
            "number": "ALT-001",
            "name": "Alternator 130A Remanufactured",
            "desc": (
                "Professional-grade remanufactured alternator with 130A "
                "output. New rectifier, voltage regulator, and bearings "
                "installed. 100% electrically tested."
            ),
            "models": "Ford Focus 2012-2018, Ford Fusion 2013-2020, Lincoln MKZ 2013-2020",
            "price": "$249.99",
            "availability": "In Stock",
        },
        {
            "number": "ST-001",
            "name": "Starter Motor 1.4kW Remanufactured",
            "desc": (
                "Heavy-duty remanufactured starter motor with 1.4kW output. "
                "New solenoid and brushes installed. Designed for cold-weather "
                "reliability. Direct bolt-on replacement."
            ),
            "models": "Volkswagen Golf 2010-2019, Audi A3 2012-2020, Seat Leon 2012-2020",
            "price": "$199.99",
            "availability": "Low Stock",
        },
        {
            "number": "IGN-001",
            "name": "Ignition Coil Pack Direct",
            "desc": (
                "High-energy direct ignition coil pack for waste-spark "
                "systems. Epoxy-sealed winding resists moisture. Improves "
                "spark energy for better combustion efficiency."
            ),
            "models": "Ford Escape 2013-2019, Ford Focus 2012-2018, Lincoln MKC 2015-2019",
            "price": "$29.99",
            "availability": "In Stock",
        },
    ],
    "Suspension & Steering": [
        {
            "number": "SHK-001",
            "name": "Front Gas Shock Absorber",
            "desc": (
                "Gas-pressurised front shock absorber for enhanced ride "
                "comfort and vehicle stability. Monotube design with "
                "multi-stage valving. Reduces body roll and nose dive."
            ),
            "models": "Nissan Sentra 2013-2019, Hyundai Elantra 2014-2020, Kia Forte 2014-2020",
            "price": "$119.99",
            "availability": "In Stock",
        },
    ],
    "Filters & Maintenance": [
        {
            "number": "CAB-001",
            "name": "Cabin Air Filter HEPA Grade",
            "desc": (
                "Premium HEPA-grade cabin air filter removes 99.97% of "
                "pollen, dust, and PM2.5 particles. Activated carbon layer "
                "neutralises odours inside the vehicle cabin."
            ),
            "models": "Toyota RAV4 2019-2023, Honda CR-V 2017-2022, Toyota Highlander 2014-2023",
            "price": "$24.99",
            "availability": "In Stock",
        },
    ],
    "Drivetrain": [
        {
            "number": "CV-001",
            "name": "Front CV Axle Shaft Assembly",
            "desc": (
                "Complete front CV axle shaft assembly with pre-packed inner "
                "and outer CV joints. New boots and clamps included. Ready to "
                "install – no additional assembly required."
            ),
            "models": "Honda Civic 2016-2022, Honda Accord 2018-2022, Acura ILX 2013-2022",
            "price": "$89.99",
            "availability": "In Stock",
        },
    ],
    "Fuel System": [
        {
            "number": "FP-001",
            "name": "Fuel Pump Module Complete",
            "desc": (
                "Complete in-tank fuel pump module assembly including fuel "
                "pump, fuel level sending unit, float arm, and strainer sock. "
                "Maintains proper fuel pressure for reliable engine operation."
            ),
            "models": "Chevrolet Malibu 2016-2022, Buick LaCrosse 2017-2019, Cadillac XT5 2017-2023",
            "price": "$149.99",
            "availability": "In Stock",
        },
    ],
}


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

def _ascii(text: str) -> str:
    """Replace common Unicode typographic characters with ASCII equivalents."""
    return (
        text.replace("\u2013", "-")   # en-dash
            .replace("\u2014", "--")  # em-dash
            .replace("\u2018", "'")   # left single quote
            .replace("\u2019", "'")   # right single quote
            .replace("\u201c", '"')   # left double quote
            .replace("\u201d", '"')   # right double quote
            .replace("\u00b0", "deg")   # degree sign
    )


def create_sample_catalog_pdf() -> None:
    """Generate a sample car parts catalog PDF using fpdf2."""
    from fpdf import FPDF  # imported here so fpdf2 is only required at setup time

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, "AutoParts Pro - Car Spare Parts Catalog 2024", ln=True, align="C")
    pdf.ln(4)

    # Introduction
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(
        0, 6,
        "Welcome to AutoParts Pro. We offer a comprehensive range of high-quality "
        "car spare parts for all major vehicle makes and models. All parts meet OEM "
        "specifications and come with a 12-month warranty.",
    )
    pdf.ln(6)

    for section, parts in CATALOG_SECTIONS.items():
        # Section header
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_fill_color(220, 220, 220)
        pdf.cell(0, 8, section, ln=True, fill=True)
        pdf.ln(2)

        for part in parts:
            # Part header
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, f"Part #: {part['number']}  |  {_ascii(part['name'])}", ln=True)

            # Details
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(0, 5, f"Description: {_ascii(part['desc'])}")
            pdf.cell(0, 5, f"Compatible Models: {_ascii(part['models'])}", ln=True)
            pdf.cell(
                0, 5,
                f"Price: {part['price']}   |   Availability: {part['availability']}",
                ln=True,
            )
            pdf.ln(3)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pdf.output(str(PDF_PATH))
    print(f"  Sample PDF catalog created at {PDF_PATH}")


# ---------------------------------------------------------------------------
# FAISS vector store
# ---------------------------------------------------------------------------

def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def init_rag(force_recreate: bool = False) -> FAISS:
    """
    Initialise the RAG system:
      1. Generate the sample PDF catalog if it does not already exist.
      2. Build (or reload) a FAISS vector store from the PDF.

    Returns the FAISS vector store ready for similarity search.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not PDF_PATH.exists():
        print("  Generating sample PDF catalog…")
        create_sample_catalog_pdf()

    if not VECTORSTORE_PATH.exists() or force_recreate:
        print("  Building FAISS vector store from PDF (requires Google API)…")
        embeddings = _get_embeddings()

        loader = PyPDFLoader(str(PDF_PATH))
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(str(VECTORSTORE_PATH))
        print(f"  Vector store saved to {VECTORSTORE_PATH}")
    else:
        embeddings = _get_embeddings()
        vectorstore = FAISS.load_local(
            str(VECTORSTORE_PATH),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    return vectorstore


def search_catalog_rag(query: str, vectorstore: FAISS) -> List[Dict[str, Any]]:
    """
    Perform a similarity search against the PDF catalog vector store and return
    the top matching document chunks.
    """
    docs = vectorstore.similarity_search(query, k=5)
    return [
        {
            "content": doc.page_content,
            "page": doc.metadata.get("page", "N/A"),
            "source": "pdf_catalog",
        }
        for doc in docs
    ]
