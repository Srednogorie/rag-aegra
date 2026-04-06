import datetime
import io
import json
import pathlib
from typing import Annotated

from fastapi import Depends, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from llama_cloud import AsyncLlamaCloud
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.readers.file import PandasCSVReader
from pydantic import BaseModel  # add to existing imports
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.db_utils import get_db_cm
from src.vector_collections import catalog_index, faq_index, manuals_index, troubleshooting_index


class DeleteFileRequest(BaseModel):
    filename: str
    category: str


ALLOWED_MIME_TYPES = {
    "application/pdf",
    "text/csv",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

app = FastAPI()

collections_map = {
    "catalog": {"index": catalog_index, "table_name": "data_techmart_catalog"},
    "faq": {"index": faq_index, "table_name": "data_techmart_faq"},
    "troubleshooting": {"index": troubleshooting_index, "table_name": "data_techmart_troubleshooting"},
    # "manuals": {"index": manuals_index, "table_name": "data_techmart_manuals"},
}


@app.get("/custom/fetch")
async def fetch_some_data(db: Session = Depends(get_db_cm)):
    result = await db.execute(text("SELECT * FROM assistant"))
    rows = result.fetchall()
    return {"rows": rows[0][2]}


@app.get("/custom/files")
async def list_files(db: Annotated[Session, Depends(get_db_cm)]):
    all_files = []
    for category, meta in collections_map.items():
        table_name = meta["table_name"]
        result = await db.execute(
            text(
                f"SELECT metadata_->>'filename' AS filename, COUNT(*) AS chunk_count "
                f"FROM public.{table_name} "
                f"WHERE metadata_::jsonb ? 'filename' "
                f"GROUP BY metadata_->>'filename'"
            )
        )
        all_files.extend([{"name": row[0], "category": category, "chunk_count": row[1]} for row in result.fetchall()])
    return {"files": all_files}



@app.post("/custom/uploadfile")
async def upload_file(file: UploadFile, category: Annotated[str, Form()] = "documents"):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    file_suffix = pathlib.Path(file.filename).suffix
    file_mime = file.content_type
    if file_mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="File type not allowed")

    file_bytes = await file.read()

    async def event_stream():
        try:
            # Step 1 processing
            yield f"data: {json.dumps({'step': 'processing', 'message': 'Processing & embedding…'})}\n\n"
            docs = []
            if file_mime == "text/csv":
                reader = PandasCSVReader(concat_rows=False, pandas_config={"header": None})
                docs = reader.load_data(io.BytesIO(file_bytes))
            elif file_mime == "application/pdf":
                # Parsing PDF documents is straightforward but spitting the result into document chunks can be
                # challenging. I experimented with all 3 parsing approaches - "text", "markdown" and "items" - see the
                # expand property of the parse method for more details.
                # The "text" mode is the easiest to use but then, unless we use more advanced spitter like
                # SemanticSplitterNodeParser, spitting only by chunk size won't be very effective and precise.
                # The "items" mode produces structured output which can be processed to create more meaningful chunks.
                # Ultimately, it all depends on our documents structure and the complexity of the content.
                client = AsyncLlamaCloud()
                # Upload and parse a document
                file_obj = await client.files.create(file=io.BytesIO(file_bytes), purpose="parse")
                result = await client.parsing.parse(
                    file_id=file_obj.id,
                    tier="agentic",
                    version="latest",
                    # Options specific to the input file type, e.g. html, spreadsheet, presentation, etc.
                    input_options={},
                    # Control the output structure and markdown styling
                    output_options={
                        "markdown": {
                            "tables": {
                                "output_tables_as_markdown": False,
                            },
                        },
                        # Saving images for later retrieval
                        # "images_to_save": ["layout"],
                    },
                    # Options for controlling how we process the document
                    processing_options={
                        "ignore": {
                            "ignore_diagonal_text": True,
                            "ignore_text_in_image": True,
                        },
                        # "ocr_parameters": {
                        #     "languages": ["fr"],
                        # }
                    },
                    page_ranges={
                        "target_pages": "5,6",
                    },
                    # Parsed content to include in the returned response
                    expand=["text"],
                )
                markdown_parser = MarkdownNodeParser()
                for page in result.markdown.pages:
                    docs = markdown_parser.get_nodes_from_documents([Document(text=page.markdown)])

                    for doc in docs:
                        doc.metadata = {
                            "filename": file.filename,
                            "filetype": file_suffix,
                            "category": category,
                            "uploaded_at": datetime.datetime.now(tz=datetime.UTC),
                            "file_mime": file_mime,
                        }
                        # collections_map[category].insert(doc)
                        print("=========================")
                        print(doc)
                        print(doc.child_nodes)
                        print(doc.parent_node)
                        print(doc.prev_node)
                        print(doc.next_node)
                        print("=========================")

            # Step 2 done
            yield f"data: {json.dumps({'step': 'done', 'message': 'Done', 'filename': file.filename})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'step': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/custom/file")
async def delete_file(body: DeleteFileRequest, db: Annotated[Session, Depends(get_db_cm)]):
    if body.category not in collections_map:
        raise HTTPException(status_code=400, detail=f"Unknown category: {body.category}")

    table_name = collections_map[body.category]["table_name"]
    await db.execute(
        text(f"DELETE FROM public.{table_name} WHERE metadata_->>'filename' = :filename"),
        {"filename": body.filename},
    )
    return {"status": "deleted"}


# The response from the parsing job in items mode - expand=["items"].

# ParsingGetResponse(
#     job=Job(
#         id='pjb-175530fgvj6f12y2uhxwq9ylgiet',
#         project_id='5b974cd3-9e79-4527-8d75-da7cfab71a2c',
#         status='COMPLETED',
#         created_at=datetime.datetime(2026, 4, 3, 15, 36, 51, 668998, tzinfo=TzInfo(0)),
#         error_message=None,
#         name='upload',
#         tier='agentic',
#         updated_at=datetime.datetime(2026, 4, 3, 15, 36, 53, 386742, tzinfo=TzInfo(0))
#     ),
#     images_content_metadata=None,
#     items=Items(
#         pages=[
#             ItemsPageStructuredResultPage(
#                 items=[
#                     HeaderItem(
#                         items=[
#                             TextItem(
#                                 md='Accessories',
#                                 value='Accessories',
#                                 bbox=[BBox(h=9.83992243680256, w=56.71081165890573, x=495.9956241961224, y=19.582252752294778, confidence=0.94, end_index=10, label='header', start_index=0)],
#                                 type='text'
#                             )
#                         ],
#                         md='Accessories',
#                         bbox=[BBox(h=9.84, w=56.71, x=496.0, y=19.58, confidence=None, end_index=None, label=None, start_index=None)],
#                         type='header'
#                     ),
#                     ImageItem(
#                         caption='Sennheiser logo',
#                         md='![Sennheiser logo](image)',
#                         url='image',
#                         bbox=[BBox(h=35.19080560946582, w=47.67403995030101, x=43.66474393760867, y=33.13128677773825, confidence=0.93, end_index=24, label='image', start_index=0)],
#                         type='image'
#                     ),
#                     HeadingItem(
#                         level=1,
#                         md='# Accessories',
#                         value='Accessories',
#                         bbox=[BBox(h=14.350997027736758, w=103.69755650738972, x=212.08240384357734, y=66.4015634102426, confidence=0.96, end_index=10, label='paragraph_title', start_index=0)],
#                         type='heading'
#                     ),
#                     TextItem(
#                         md='A variety of accessories are available for the XSW IEM series.',
#                         value='A variety of accessories are available for the XSW IEM series.',
#                         bbox=[BBox(h=12.175479945261879, w=339.26409009012366, x=212.62548027001358, y=95.29720329493082, confidence=0.96, end_index=61, label='text', start_index=0)],
#                         type='text'
#                     ),
#                     HeadingItem(
#                         level=2,
#                         md='## Earphones',
#                         value='Earphones',
#                         bbox=[BBox(h=17.252300583109044, w=89.60335677728419, x=213.09225802091274, y=131.115271121588, confidence=0.96, end_index=8, label='paragraph_title', start_index=0)],
#                         type='heading'
#                     ),
#                     TextItem(
#                         md='**IE 4**\nArt. no. 500432',
#                         value='IE 4\nArt. no. 500432',
#                         bbox=[BBox(h=35.299738438155, w=85.65634919552681, x=341.0364702178211, y=170.35442570845441, confidence=0.92, end_index=23, label='caption', start_index=0)],
#                         type='text'
#                     ),
#                     ImageItem(
#                         caption='IE 4 earphones with 3.5mm jack',
#                         md='![IE 4 earphones with 3.5mm jack](page_5_image_1_v2.jpg)',
#                         url='page_5_image_1_v2.jpg',
#                         bbox=[BBox(h=185.25405040908447, w=144.33008385337266, x=316.2370080268674, y=242.53847465939407, confidence=0.99, end_index=55, label='image', start_index=0)],
#                         type='image'
#                     ),
#                     TextItem(
#                         md='Earphones for use with wireless monitor systems. The IE 4 feature excellent sound and dynamic range. They have interchangeable earpieces to fit different sized ear canals. This provides good insulation against background noise and exceptionally good bass response for this type of receiver.',
#                         value='Earphones for use with wireless monitor systems. The IE 4 feature excellent sound and dynamic range. They have interchangeable earpieces to fit different sized ear canals. This provides good insulation against background noise and exceptionally good bass response for this type of receiver.',
#                         bbox=[BBox(h=68.1033243610661, w=340.66754263826704, x=212.28946480206744, y=443.07694738054977, confidence=0.99, end_index=289, label='text', start_index=0)],
#                         type='text'
#                     ),
#                     TextItem(
#                         md='Removable earpieces in three different sizes for universal fit',
#                         value='Removable earpieces in three different sizes for universal fit',
#                         bbox=[BBox(h=11.931921844035852, w=332.6062198099276, x=212.30683360625477, y=519.9985701809023, confidence=0.95, end_index=61, label='text', start_index=0)],
#                         type='text'
#                     ),
#                     ListItem(
#                         items=[
#                             TextItem(
#                                 md='Natural sound and dynamic range',
#                                 value='Natural sound and dynamic range',
#                                 bbox=[BBox(h=11.760622174518897, w=200.8115564039277, x=212.22169101919778, y=541.231908671756, confidence=0.97, end_index=30, label='text', start_index=0)],
#                                 type='text'
#                             ),
#                             TextItem(
#                                 md='Extremely resilient at high sound pressures',
#                                 value='Extremely resilient at high sound pressures',
#                                 bbox=[BBox(h=11.812408114996236, w=253.7701746602873, x=212.30105582037206, y=558.4390281523914, confidence=0.97, end_index=42, label='text', start_index=0)],
#                                 type='text'
#                             ),
#                             TextItem(
#                                 md='Superior bass response',
#                                 value='Superior bass response',
#                                 bbox=[BBox(h=12.114601386242352, w=144.2871049246439, x=212.42269061949196, y=575.0427637224896, confidence=0.97, end_index=21, label='text', start_index=0)],
#                                 type='text'
#                             ),
#                             TextItem(
#                                 md='Sturdy cable',
#                                 value='Sturdy cable',
#                                 bbox=[BBox(h=11.83697259789451, w=83.41761667540015, x=212.52693889115497, y=592.263619260909, confidence=0.97, end_index=11, label='text', start_index=0)],
#                                 type='text'
#                             )
#                         ],
#                         md='* Natural sound and dynamic range\n\n* Extremely resilient at high sound pressures\n\n* Superior bass response\n\n* Sturdy cable',
#                         ordered=False,
#                         bbox=[BBox(h=62.87, w=253.85, x=212.22, y=541.23, confidence=0.97, end_index=121, label='list', start_index=0)],
#                         type='list'
#                     ),
#                     FooterItem(
#                         items=[
#                             TextItem(
#                                 md='SENNHEISER\n4',
#                                 value='SENNHEISER\n4',
#                                 bbox=[BBox(h=9.88269252274673, w=81.23444217458585, x=44.23273928107285, y=798.4370840957271, confidence=0.94, end_index=11, label='footer', start_index=0)],
#                                 type='text'
#                             )
#                         ],
#                         md='SENNHEISER\n4',
#                         bbox=[BBox(h=9.88, w=81.23, x=44.23, y=798.44, confidence=None, end_index=None, label=None, start_index=None)],
#                         type='footer'
#                     )
#                 ],
#                 page_height=841.89,
#                 page_number=5,
#                 page_width=595.275,
#                 success=True
#             ),
#             ItemsPageStructuredResultPage(
#                 items=[
#                     HeaderItem(
#                         items=[
#                             TextItem(
#                                 md='Earphones',
#                                 value='Earphones',
#                                 bbox=[BBox(h=10.038717721073803, w=49.24116246218799, x=503.3574450207222, y=19.67432099817788, confidence=0.94, end_index=8, label='header', start_index=0)],
#                                 type='text'
#                             )
#                         ],
#                         md='Earphones',
#                         bbox=[BBox(h=10.04, w=49.24, x=503.36, y=19.67, confidence=None, end_index=None, label=None, start_index=None)],
#                         type='header'
#                     ),
#                     ImageItem(
#                         caption='Sennheiser Logo',
#                         md='![Sennheiser Logo](image)',
#                         url='image',
#                         bbox=[BBox(h=34.881409054267706, w=47.13701602042594, x=44.03258153608369, y=33.24461865413015, confidence=0.96, end_index=24, label='image', start_index=0)],
#                         type='image'
#                     ),
#                     HeadingItem(
#                         level=1,
#                         md='# Other compatible earphones',
#                         value='Other compatible earphones',
#                         bbox=[BBox(h=12.433087426757814, w=166.2701800700862, x=212.25802372183452, y=66.15194565213832, confidence=0.87, end_index=25, label='paragraph_title', start_index=0)],
#                         type='heading'
#                     ),
#                     HeadingItem(
#                         level=2,
#                         md='## IE 100 PRO',
#                         value='IE 100 PRO',
#                         bbox=[BBox(h=11.470623414425726, w=70.49949766801046, x=212.55332883955793, y=97.56536357005051, confidence=0.83, end_index=9, label='text', start_index=0)],
#                         type='heading'
#                     ),
#                     ImageItem(
#                         caption='IE 100 PRO red in-ear monitor with black cable and ear tip',
#                         md='![IE 100 PRO red in-ear monitor with black cable and ear tip](page_6_image_1_v2.jpg)',
#                         url='page_6_image_1_v2.jpg',
#                         bbox=[BBox(h=173.36840002977326, w=190.44303606507836, x=290.71994837393413, y=151.78764956650153, confidence=0.99, end_index=83, label='image', start_index=0)],
#                         type='image'
#                     ),
#                     TextItem(
#                         md='For the demanding requirements of the live stage: With a newly developed dynamic driver, the IE 100 PRO guarantees precise acoustic reproduction for live sessions and sets. The innovative new membrane delivers powerful, warm and detail-rich sound. Every detail remains undistorted and clear, even at the highest levels. Because of its exceptional sound and unmatched comfort, musicians and DJs choose the IE 100 PRO for live sessions, for producing and for everyday listening.', value='For the demanding requirements of the live stage: With a newly developed dynamic driver, the IE 100 PRO guarantees precise acoustic reproduction for live sessions and sets. The innovative new membrane delivers powerful, warm and detail-rich sound. Every detail remains undistorted and clear, even at the highest levels. Because of its exceptional sound and unmatched comfort, musicians and DJs choose the IE 100 PRO for live sessions, for producing and for everyday listening.', bbox=[BBox(h=109.95801986322172, w=340.84812503200624, x=212.14241483437144, y=354.11870289425735, confidence=0.99, end_index=475, label='text', start_index=0)],
#                         type='text'
#                     ),
#                     TextItem(
#                         md='The in-ear earphones fit any ear shape. Their low-profile, compact design ensures a secure fit and unmatched comfort. They have a sturdy, stage-proof design, from the connector to the cable sheath.',
#                         value='The in-ear earphones fit any ear shape. Their low-profile, compact design ensures a secure fit and unmatched comfort. They have a sturdy, stage-proof design, from the connector to the cable sheath.',
#                         bbox=[BBox(h=52.8605112751286, w=341.07586904609497, x=212.19927108318052, y=493.9699939703221, confidence=0.99, end_index=196, label='text', start_index=0)],
#                         type='text'
#                     ),
#                     ListItem(
#                         items=[
#                             TextItem(
#                                 md='Newly developed 10 mm dynamic wideband transducer for powerful, accurate monitoring sound',
#                                 value='Newly developed 10 mm dynamic wideband transducer for powerful, accurate monitoring sound',
#                                 bbox=[BBox(h=26.718737916825468, w=338.8242517089844, x=212.20283346036584, y=556.6811636772343, confidence=0.98, end_index=88, label='text', start_index=0)],
#                                 type='text'
#                             ),
#                             TextItem(
#                                 md='Dynamic driver system delivers homogeneous and distortion-free sound for less strain on the listener',
#                                 value='Dynamic driver system delivers homogeneous and distortion-free sound for less strain on the listener',
#                                 bbox=[BBox(h=26.409425958698588, w=333.1382900859554, x=212.25205097998642, y=587.6781822777725, confidence=0.98, end_index=99, label='text', start_index=0)],
#                                 type='text'
#                             ),
#                             TextItem(
#                                 md='Secure fit, unmatched comfort: new low-profile and ergonomic design',
#                                 value='Secure fit, unmatched comfort: new low-profile and ergonomic design',
#                                 bbox=[BBox(h=26.59812129263519, w=332.436554950249, x=212.48245348451195, y=618.5154816078746, confidence=0.98, end_index=66, label='text', start_index=0)],
#                                 type='text'
#                             ),
#                             TextItem(
#                                 md='Optimized earpiece shape and flexible silicone and foam attachments for excellent protection against background noise',
#                                 value='Optimized earpiece shape and flexible silicone and foam attachments for excellent protection against background noise',
#                                 bbox=[BBox(h=38.908386304901875, w=340.7248242455459, x=212.30296993348657, y=649.8176512439263, confidence=0.98, end_index=116, label='text', start_index=0)],
#                                 type='text'
#                             ),
#                             TextItem(
#                                 md='Stage-proof cable connection',
#                                 value='Stage-proof cable connection',
#                                 bbox=[BBox(h=12.218173267197031, w=177.60183604766104, x=212.47878476770913, y=694.8994430223419, confidence=0.97, end_index=27, label='text', start_index=0)],
#                                 type='text'
#                             )
#                         ],
#                         md='* Newly developed 10 mm dynamic wideband transducer for powerful, accurate monitoring sound\n* Dynamic driver system delivers homogeneous and distortion-free sound for less strain on the listener\n* Secure fit, unmatched comfort: new low-profile and ergonomic design\n* Optimized earpiece shape and flexible silicone and foam attachments for excellent protection against background noise\n* Stage-proof cable connection',
#                         ordered=False,
#                         bbox=[BBox(h=150.44, w=340.82, x=212.2, y=556.68, confidence=0.98, end_index=414, label='list', start_index=0)],
#                         type='list'
#                     ),
#                     FooterItem(
#                         items=[
#                             TextItem(
#                                 md='SENNHEISER 5',
#                                 value='SENNHEISER 5',
#                                 bbox=[BBox(h=9.90154200350139, w=81.26711945287192, x=44.18799245620355, y=798.418134351777, confidence=0.94, end_index=11, label='footer', start_index=0)],
#                                 type='text'
#                             )
#                         ],
#                         md='SENNHEISER 5',
#                         bbox=[BBox(h=9.9, w=81.27, x=44.19, y=798.42, confidence=None, end_index=None, label=None, start_index=None)],
#                         type='footer'
#                     )
#                 ],
#                 page_height=841.89,
#                 page_number=6,
#                 page_width=595.275,
#                 success=True
#             )
#         ]
#     ),
#     job_metadata=None,
#     markdown=None,
#     markdown_full=None,
#     metadata=None,
#     raw_parameters=None,
#     result_content_metadata=None,
#     text=None,
#     text_full=None
# )
