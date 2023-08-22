from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy import MetaData, Table, String, Integer, Column, DateTime, Boolean, Double
from sqlalchemy.orm import declarative_base

import dotenv

Base = declarative_base()

class ServiceHistory(Base):
    __tablename__ = 'service_history'
    id = Column('id', String(36), primary_key=True)
    user_id = Column('used_id', Integer(), nullable=True)
    filename = Column('filename', String(200), nullable=False)
    datetime = Column('datetime', DateTime(), nullable=False)
    state = Column('state', String(100), nullable=False)
    vt1 = Column('vt1', Boolean(), nullable=True)
    vt2 = Column('vt2', Boolean(), nullable=True)
    vt3 = Column('vt3', Boolean(), nullable=True)
    vt4 = Column('vt4', Boolean(), nullable=True)
    vt5 = Column('vt5', Boolean(), nullable=True)
    vt6 = Column('vt6', Boolean(), nullable=True)
    vt7 = Column('vt7', Boolean(), nullable=True)
    vt1_conf = Column('vt1_conf', Double(), nullable=True)
    vt2_conf = Column('vt2_conf', Double(), nullable=True)
    vt3_conf = Column('vt3_conf', Double(), nullable=True)
    vt4_conf = Column('vt4_conf', Double(), nullable=True)
    vt5_conf = Column('vt5_conf', Double(), nullable=True)
    vt6_conf = Column('vt6_conf', Double(), nullable=True)
    vt7_conf = Column('vt7_conf', Double(), nullable=True)
    pixel_spacing_w = Column('pixel_spacing_w', Double(), nullable = True)
    pixel_spacing_h = Column('pixel_spacing_h', Double(), nullable = True)
    slice_thickness = Column('slice_thickness', Double(), nullable = True)

def create_table(engine, session):

    try:
        Base.metadata.drop_all(engine)
    except:
        pass

    Base.metadata.create_all(engine)
    
    print(session.query(ServiceHistory).all())
    
def update_value(session, id, value):
    session.query(ServiceHistory).filter(ServiceHistory.id == id).update(
        {ServiceHistory.state : value}, synchronize_session = False
    )
    session.commit()
    
def send_results_to_db(session, id, results):
    session.query(ServiceHistory).filter(ServiceHistory.id == id).update(
        {eval(f'ServiceHistory.vt{k}') : v[0] for k, v in results.items()}, synchronize_session = False
    )
    session.query(ServiceHistory).filter(ServiceHistory.id == id).update(
        {eval(f'ServiceHistory.vt{k}_conf') : v[1] for k, v in results.items()}, synchronize_session = False
    )
    session.commit()
    
def update_spacing(session, fileid, params):
    session.query(ServiceHistory).filter(ServiceHistory.id == fileid).update(
        {'pixel_spacing_w' : params[0],
         'pixel_spacing_h' : params[1],
         'slice_thickness' : params[2]},
        synchronize_session = False
    )
    session.commit()
        
if __name__ == '__main__':
    config = dotenv.dotenv_values('.env')
    engine = create_engine(f"postgresql+psycopg2://{config['AWS_RDS_USERNAME']}:{config['AWS_RDS_PASSWORD']}@{config['AWS_RDS_ENDPOINT']}:{config['AWS_RDS_PORT']}/{config['AWS_RDS_DB_NAME']}")
    engine.connect()
    session = Session(engine)
    
    #create_table(engine, session)
    #update_value(engine, session)
    #print(ServiceHistory.__getattribute__(ServiceHistory, 'vt1'))
    # send_results_to_db(session, '3b4958ed-4c34-460c-841d-8cfc07fbc2a5', {
    #     '1' : [0, 0.45]
    # })