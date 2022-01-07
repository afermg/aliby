from core.experiment import Experiment
from core.segment import Tiler
expt = Experiment.from_source(19310, #Experiment ID on OMERO
                                'upload', #OMERO Username
                                '***REMOVED***', #OMERO Password
                                'islay.bio.ed.ac.uk', #OMERO host
                                port=4064 #This is default
                                )


# Load whole position
img=expt[0,0,:,:,2]
plt.imshow(img[0,0,...,0]); plt.show()

# Manually get template
tilesize=117
x0=827
y0=632
trap_template = img[0,0,x0:x0+tilesize,y0:y0+tilesize,0]
plt.imshow(trap_template); plt.show()

tiler = Tiler(expt, template = trap_template)

# Load images (takes about 5 mins)
trap_tps = tiler.get_traps_timepoint(0, tile_size=117, z=[2])

#Plot found traps
nrows, ncols = (5,5)
fig, axes = plt.subplots(nrows,ncols)
for i in range(nrows):
    for j in range(ncols):
        if i*nrows+j < trap_tps.shape[0]:
            axes[i,j].imshow(trap_tps[i*nrows+j,0,0,...,0])
plt.show()

