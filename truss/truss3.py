from matplotlib import pyplot as plt

fig = plt.figure()
axes = fig.add_axes([0.1,0.1,2,2])
fig.gca().set_aspect('equal', adjustable='box')

#Plot members
for n, mbr in enumerate(members):
    node_i = mbr[0] #Node number for node i of this member
    node_j = mbr[1] #Node number for node j of this member

    ix = nodes[node_i-1,0] #x-coord of node i of this member
    iy = nodes[node_i-1,1] #y-coord of node i of this member
    jx = nodes[node_j-1,0] #x-coord of node j of this member
    jy = nodes[node_j-1,1] #y-coord of node j of this member

    if(abs(mbrForces[n])<0.001):
        axes.plot([ix,jx],[iy,jy],'grey',linestyle='--') #Zero force in member
    elif(mbrForces[n]>0):
        axes.plot([ix,jx],[iy,jy],'b') #Member in tension
    else:
        axes.plot([ix,jx],[iy,jy],'r') #Member in compression

axes.set_xlabel('Distance (m)')
axes.set_ylabel('Distance (m)')
axes.set_title('Tension/compression members')
axes.grid()
plt.show()
