import pandas as pd
from sqlalchemy import create_engine
from database.records import Base, Position, Trap, Cell
from sqlalchemy.orm import sessionmaker

engine = create_engine("sqlite:///:memory:", echo=False)

# Create the necessary tables
Base.metadata.create_all(engine)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Create a new position
pos1 = Position(name="pos001", n_timepoints=0)
print("Created position: ", pos1)

session.add(pos1)

# Create a new trap
trap1 = Trap(position=pos1, number=1, x=1, y=10, size=96)
print("Created trap: ",trap1)

session.add(trap1)

session.commit()

# Query session for pos1
queried_position = session.query(Position).filter_by(name='pos001').first()
print("Queried for position: ", pos1)
print("Success? ", queried_position is pos1)
print("Traps added to this posistion: ",queried_position.traps)

# Create a new cell
cell1 = Cell(number=1, trap=trap1)
cell2 = Cell(number=2, trap=trap1)
session.add_all([cell1, cell2])
print("Creating cells: ", cell1, cell2)

print("Cells in trap1: ")
print(trap1.cells)


# Update the number of time points in the positions
queried_position.n_timepoints += 1
session.commit()

# Check how it works with Panda
positions = pd.read_sql_table('positions', engine)
print(positions)
traps = pd.read_sql_table('traps', engine)
print(traps)
cells = pd.read_sql_table('cells', engine)
print(cells)

# Try a joined query

queried_cell = session.query(Cell).filter_by(number=2)\
                .join(Trap).filter_by(number=1)\
                .join(Position).filter_by(name="pos001")\
                .first()

print(queried_cell)
