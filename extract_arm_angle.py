#%%
import numpy as np

x = np.load("data.npy")

# %%
j = []
for k in x:
    m = []
    for l in k[5:11]:
        m.append(l[:2])
    j.append(m)
j = np.array(j)
# %%
def findAngleR(p0,p1,p2):
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)
    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1)) 
    return np.degrees(angle)%360

def findAngleL(p0,p1,p2):
    v0 = np.array(p0) - np.array(p1)
    v1 = np.array(p2) - np.array(p1)
    angle = np.math.atan2(np.linalg.det([v0,v1]),np.dot(v0,v1)) 
    return 360-np.degrees(angle)%360

#%%
X_angle = []
Y_angle = []
AngleA = np.zeros((50,4))
for i in range(len(j)):
    c = j[i]
    AngleA[i][0] = (findAngleR(c[3],c[4],c[5]))
    AngleA[i][1] = (findAngleR(c[4],c[3],c[0]))
    AngleA[i][2]= (findAngleL(c[1],c[0],c[3]))
    AngleA[i][3] = (findAngleL(c[0],c[1],c[2]))
X_angle.append(AngleA)
X_angle = np.array(X_angle)
print(X_angle.shape)


# for B in new_Y:


#     AngleB = np.zeros((4,300))
#     for i in range(len(B)):
#         c = B[i]
#         AngleB[0][i] = (findAngleR(c[3],c[4],c[5]))
#         AngleB[1][i] = (findAngleR(c[4],c[3],c[0]))
#         AngleB[2][i] = (findAngleL(c[1],c[0],c[3]))
#         AngleB[3][i] = (findAngleL(c[0],c[1],c[2]))
#     Y_angle.append(AngleB)
# Y_angle = np.array(Y_angle)
# print(Y_angle.shape)

# %%
