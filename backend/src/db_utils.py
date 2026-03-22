import os

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

db_user = os.getenv("POSTGRES_USER")
db_password = os.getenv("POSTGRES_PASSWORD")
db_address = os.getenv("POSTGRES_HOST")
db_port = os.getenv("POSTGRES_PORT")
db_database = os.getenv("POSTGRES_VECTOR_DB")
DATABASE_URL = (
    f"postgresql+asyncpg://{db_user}:{db_password}"
    f"@{db_address}:{db_port}/{db_database}"
)

engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)

async def get_dbb():
    db = AsyncSession(engine)
    try:
        yield db
    finally:
        await db.close()

async def get_db():
    async with async_session() as session:
        yield session


async def get_db_cm():
    async with async_session() as session, session.begin():
        yield session
