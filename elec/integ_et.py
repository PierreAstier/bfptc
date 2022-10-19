from __future__ import print_function
import numpy as np


def ECoulomb(X,X_q) :
    """
    X = where, X_q = charge location.
    both should be numpy arrays.
    if X is multi-d, the routine assumes that the 
    physical coordinates (x,y,z) are patrolled by 
    the last index
    """
    d = X-X_q
    r3 = np.power((d**2).sum(axis=-1), 1.5)
    # anything more clever ?
    # of course d/r3 does not work 
    return (d.T / r3.T).T



class CcdGeom() :
    def __init__(self, z_q, zsh, zsv, a, b, thickness, pixsize) :
        """
        parameters :  (all in microns)
        z_q : altitude of the burried channel (microns)
        zsh : vertex altitude for horizontal boundaries
        zsv : vertex altitude for vertical boundaries
        a, b : size of the rectangular charge source 
        thickness : thickness
        pixsize : pixel size
        """
        # z of the charge (distance to clock rails)
        self.z_q = z_q
        # height of vertex for horizontal boundaries
        self.zsh = zsh
        # for vertical boundaries
        self.zsv = zsv
        self.b = np.fabs(b)
        self.a = np.fabs(a)
        # overall thickness
        self.t = np.fabs(float(thickness))
        # pixel size
        self.pix = float(pixsize)
        #
        self.nstepz = 100
        self.nstepxy = 20
        self.charge_split=3

    def ECoulombChargeSheet(self,X, X_q) :
        """
        X = where (the last index should address x,y,z.
        X_q = charge location
        Both Should be numpy arrays.
        if X is multi-d, the routine assumes
        that the physical coordinates (x,y,z) are patrolled by the last index.
        Returns the electric field from a unitely charged horizontal rectangle
        centered at X_q of size 2a * 2b.  
        """
        # use Durand page 244 tome 1
        # four corners : 
        X1 = X_q + np.array([ self.a, self.b,0])
        X2 = X_q + np.array([-self.a, self.b,0])
        X3 = X_q + np.array([-self.a,-self.b,0])
        X4 = X_q + np.array([ self.a,-self.b,0])

        # distances to the four corners
        d1 = np.sqrt(((X-X1)**2).sum(axis=-1))
        d2 = np.sqrt(((X-X2)**2).sum(axis=-1))
        d3 = np.sqrt(((X-X3)**2).sum(axis=-1))
        d4 = np.sqrt(((X-X4)**2).sum(axis=-1))
        # reserve the returned array
        ret = np.ndarray(X.shape)
        x = X[...,0]
        y = X[...,1]
        if False :  # old debug
            deno = (d3+y+self.b)*(d1+y-self.b)
            ind = (deno==0)
            if ind.sum() != 0:
                ind = np.where(ind)[0][0]
                print('singular deno, b=%f'%self.b, 'd1=%f d3=%f y=%f '%(d1[ind], d3[ind], y[ind]))
                print(' X ',X[ind], 'num ',((d4+y+self.b)*(d2+y-self.b))[ind])
        # Ex
        # note : if a or b goes to 0, the log is 0 and the denominator (last
        # line) is zero as well. So some expansion would be required
        ret[...,0] = np.log((d4+y+self.b)*(d2+y-self.b)
                            /(d3+y+self.b)/(d1+y-self.b))
        # Ey
        ret[...,1] = np.log((d2+x+self.a)*(d4+x-self.a)
                            /(d3+x+self.a)/(d1+x-self.a))
        # Ez is almost never used here and horrible (p 246). 
        # use the point source approximation 
        ret[...,2] = (4*self.a*self.b)*ECoulomb(X,X_q)[...,2]
        #
        return ret/(4*self.a*self.b)


    def IntegrateAlongZ(self, X, Ex_or_Ey, zstart, zend, npair=11) :
        """
    
        Integrate transverse E Field along Z at point X (2 coordinates, last
        index).  The coordinate of the field is given by Ex_or_Ey (0,
        or 1).  at point X from the point charge The computation uses
        the dipole series trick. The number of dipoles is an optional
        argument. Odd numbers are better for what we are doing here.

        """
        # The integral of the field (x_or_y/r^3 dz from z1 to z2) reads
        # x_or_y/rho**2*(z2/r2-z1/r1) with rho2 = x**2+y**2
        # x_or_y/rho2 does not change when going through image sources
        # so we use them as arguments, dz1 and dz2 z{begin,end}--Xq[2]
        # just for test: if zstart==zend, then return the field value
        if zstart != zend  :
            def integral(rho2, x_or_y, dz1, dz2) :
                r1 = np.sqrt(rho2+dz1**2)
                r2 = np.sqrt(rho2+dz2**2)
                return x_or_y*(dz2/r2- dz1/r1)/rho2
        else :
            def integral(rho2, x_or_y, dz1, dz2) :
                """
                x_or_y/r**3
                """
                r = np.sqrt(rho2+dz1**2)
                vals =  x_or_y/r**3
                return vals

        # reserve the result array       
        result = np.zeros(X.shape[:-1])
        assert (Ex_or_Ey ==0) or (Ex_or_Ey ==1),"IntegrateAlongZ : Ex_or_Ey should be 0 or 1"
        zqp = self.z_q
        zqm = -zqp
                
        # for the first dipole, generate a set of point charges to emulate
        # an extended distribution (size 2a*2b)
        xstep = 2*self.a/self.charge_split
        ystep = 2*self.b/self.charge_split
        xqpos =  -self.a+(np.linspace(0,self.charge_split-1, self.charge_split)+0.5)*xstep
        yqpos =  -self.b+(np.linspace(0,self.charge_split-1, self.charge_split)+0.5)*ystep
        # print('xqpos, yqpos', xqpos, yqpos)
        for xq in xqpos:
            for yq in yqpos :
                dx = X[...,0]-xq
                dy = X[...,1]-yq
                dX = [dx,dy]
                rho2 = dx**2+dy**2
                result += integral(rho2, dX[Ex_or_Ey], zstart-zqp, zend-zqp)
                # image charge, switch sign of z and q
                result -= integral(rho2, dX[Ex_or_Ey], zstart-zqm, zend-zqm)
        result /= self.charge_split*self.charge_split

            
        # next dipoles : no more extended charge
        # The (x,y) charge coordinates are 0, and common to all images:
        rho2 = X[...,0]**2+X[...,1]**2
        x_or_y = X[...,Ex_or_Ey]
        for i in range(1, npair) : 
            if (i%2):
                ztmp = 2*self.t-zqm
                zqm = 2*self.t-zqp
                zqp = ztmp
            else :
                ztmp = -zqm
                zqm = -zqp
                zqp = ztmp
            result += integral(rho2, x_or_y, zstart-zqp, zend-zqp)
            result -= integral(rho2, x_or_y, zstart-zqm, zend-zqm)

        # 55 = 8.85418781e-12 (F/m) *1e-6 (microns/m)  / 1.602e-19 (Coulomb/electron)
        # eps_r_Si = 12, so eps = 55*12 = 660 el/V/um
        # This routine hence returns the field sourced by -1 electron
        result *= 1/(4*3.1415927*660)
        return result


    

    def Exyz(self, X, npair=11) :
        """
        Field at point X from the point charge 
        if X is multi-dimensional, x,y,z should be represented 
        by the last index ([0:3]). 
        The computation uses the dipole series trick. The number of dipoles is an
        optional argument. Odd numbers are better for what we are
        doing here.
        """
        # put the center of the aggressor pixel at x,y, = 0,0
        # this assumption is relied on in Eval_Eth and Eval_Etv
        qpos1=np.array([0,0,self.z_q])
        # split the calculation in 2 parts: approximation when far from the source, image method when near.
        rho = np.sqrt(X[...,0]**2+X[...,1]**2)
        index_close = rho/self.t<2 # this is the separating value. 
        X_close = X[index_close]
        # image charge w.r.t. the parallel clock lines
        qpos2=np.array([qpos1[0], qpos1[1], -qpos1[2]])
        # first dipole
        E_close = self.ECoulombChargeSheet(X_close, qpos1) - self.ECoulombChargeSheet(X_close, qpos2)
        # next dipoles
        for i in range(1, npair) : 
            if (i%2):
                qpos1[2] = 2*self.t-qpos1[2]
                qpos2[2] = 2*self.t-qpos2[2]
                E_close += ECoulomb(X_close,qpos2)- ECoulomb(X_close,qpos1)
            else :
                qpos1[2] = -qpos1[2]
                qpos2[2] = -qpos2[2]
                E_close += ECoulomb(X_close,qpos1)- ECoulomb(X_close,qpos2)
        X_far = X[~index_close]
        rho_far = rho[~index_close]
        # Jon Pumplin, Am. Jour. Phys. 37,7 (1969), eq 7
        # When changing coordinate system (shift along z), cos -> sin.
        # And since this only applies far from the source, the latter
        # can be regarded as point-like.
        # I checked the continuity over the separation point.
        fact = -np.sin(np.pi*self.z_q/self.t) * np.sin(np.pi*X_far[...,2]/self.t)*np.exp(-np.pi*rho_far/self.t)*np.sqrt(8/rho_far/self.t)*(-np.pi/self.t-0.5/rho_far)
        E_far = np.zeros_like(X_far)
        E_far[...,0] = X_far[...,0]/rho_far*fact
        E_far[...,1] = X_far[...,1]/rho_far*fact
        # aggregate the results
        E = np.zeros_like(X)
        E[index_close] = E_close
        E[~index_close] = E_far
        # epsilon0 = 55 el/V/micron 
        # 55 = 8.85418781e-12 (F/m) *1e-6 (microns/m)  / 1.602e-19 (Coulomb/electron)
        # eps_r_Si = 12, so eps = 55*12 = 660 el/V/um
        # This routine hence returns the field sourced by -1 electron
        E *= 1/(4*3.1415927*660)
        return E


    def Integ_Ey_fast(self, i, j, top_or_bottom, z_end=None, npair=11):
        """
        X is a 2 ccordinate vector, x and y
        return the integral of Ex along z from self.zsh to zend.
        """
        z_end = self.t if (z_end==None) else z_end
        assert z_end > self.zsh
        xystep = self.pix/(self.nstepxy)
        xx = (i-0.5)*self.pix + (np.linspace(0,self.nstepxy-1,self.nstepxy)+0.5)*xystep
        yy = np.ones(xx.shape)*(j+0.5*top_or_bottom)*self.pix
        X = np.array([xx,yy]).T
        # by definition of zsh, we  integrate from zsh to z_end,
        # and divide by the pixel size to be consistent with Eval_ET{v,h}
        return top_or_bottom*self.IntegrateAlongZ(X, 1, self.zsh, z_end, npair=npair).mean()/self.pix

    def Eval_ETh(self, i,j, top_or_bottom, z_end=None):
        """
        Returns the field transverse to the horizontal pixel boundary.
        return a 2d array of shifts at evenly spaced points in x and z.
        normalized in units of pixel size, for a unit charge.
        The returned array has 3 indices: 
        [along the pixel side, along the drift,E-field coordinate].
        The resutl is multiplied by the z- and x- steps, so that the
        sum is the averge over x, divided by the pixel size. 
        """
        assert np.abs(top_or_bottom) == 1
        z_end = self.t if (z_end==None) else z_end
        # by definition of zsh, we  integrate from zsh to z_end:
        zstep = (z_end-self.zsh)/(self.nstepz)
        xystep = self.pix/(self.nstepxy)
        z = self.zsh+(np.linspace(0, self.nstepz-1, num = self.nstepz)+0.5)*zstep
        x = (i-0.5)*self.pix + (np.linspace(0,self.nstepxy-1,self.nstepxy)+0.5)*xystep
        [xx,zz] = np.meshgrid(x,z)
        yy = np.ones(xx.shape)*(j+0.5*top_or_bottom)*self.pix
        X=np.array([xx, yy, zz]).T
        return self.Exyz(X)*zstep*xystep

    
    def average_shift_h(self, i,j, top_or_bottom, z_end=None):
        """
        Integrate the field transverse to the horizontal pixel boundary
        """
        sum = self.Eval_ETh(i,j,top_or_bottom,z_end)[...,1].sum() # select Ey
        # we want the integral over z and the average over x,
        # divided by the pixel size, with a sign that defines
        # if in moves inside or outside. Here is it:
        return sum*(top_or_bottom/(self.pix**2))

    def corner_shift_h(self, i,j, top_or_bottom, z_end=None):
        E = self.Eval_ETh(i,j,top_or_bottom, z_end)[...,1]
        # integrate over z
        #(multiplication by the step done in the calling routine)
        intz = E.sum(axis=1)
        x = range(intz.shape[0])
        p = np.polyfit(x, intz, 1)
        return p[0]*-0.5+p[1], p[0]*(x[-1]+0.5)+p[1]
    
    def Eval_ETv(self, i,j, left_or_right, z_end=None):
        """
        Returns the field transverse to the vertical pixel boundary.
        return a 3d array of shifts at evenly spaced points in y and z.
        Ex,y,z is indexed by the last index.
        If you are only interested in the boundary shift, use average_shift_{h,v}
        """
        assert np.abs(left_or_right) == 1
        z_end = self.t if (z_end==None) else z_end
        # the source charge is at x,y=0
        zstep = (z_end-self.zsv)/(self.nstepz)
        xystep = self.pix/(self.nstepxy)
        z = self.zsv+(np.linspace(0,self.nstepz-1,self.nstepz)+0.5)*zstep
        y = (j-0.5)*self.pix + (np.linspace(0,self.nstepxy-1,self.nstepxy)+0.5)*xystep
        [yy,zz] = np.meshgrid(y,z)
        xx = np.ones(yy.shape)*(i+0.5*left_or_right)*self.pix
        X=np.array([xx, yy, zz]).T
        return self.Exyz(X)*zstep*xystep

    def Integ_Ex_fast(self, i,j,left_or_right, z_end=None, npair=11):
        """
        X is a 2 ccordinate vector, x and y
        return the integral of Ex along z from self.zsh to zend.
        """
        z_end = self.t if (z_end==None) else z_end
        assert z_end > self.zsv
        xystep = self.pix/(self.nstepxy)
        yy = (j-0.5)*self.pix + (np.linspace(0,self.nstepxy-1,self.nstepxy)+0.5)*xystep
        xx = np.ones(yy.shape)*(i+0.5*left_or_right)*self.pix
        X = np.array([xx,yy]).T
        # by definition of zsh, we  integrate from zsv to z_end,
        # and divide by the pixel size to be consistent with Eval_ET{v,h}
        return left_or_right*self.IntegrateAlongZ(X, 0, self.zsv, z_end, npair=npair).mean()/self.pix
    

    
    def average_shift_v(self, i,j, left_or_right, z_end=None):
        """
        Average shift of the vertical boundary of pixel (i j).
        """
        sum = self.Eval_ETv(i,j,left_or_right, z_end)[...,0].sum() # select Ex
        # we want the integral over z and the average over x,
        # divided by the pixel size, with a sign that defines
        # if in moves inside or outside. Here is it:
        return sum*(left_or_right/(self.pix**2))

    def dx_dy(self, imax, z_end=None):
        """
        corner shifts calculations.  The returned array are larger by 1
        than imax, because there are more corners than pixels
        """
        last_i = imax
        shifts_h = np.ndarray((last_i, last_i))
        shifts_v = np.ndarray((last_i, last_i))
        for i in range(last_i):
            for j in range(last_i):
                shifts_h[i,j] = self.average_shift_h(i+0.5,j,+1, z_end)
                shifts_v[i,j] = self.average_shift_v(i,j+0.5,+1, z_end)
        # parametrize the corner shifts of imax pixels: imax+1
        # corners in each direction
        dx  = np.zeros((imax+1,imax+1))
        dy  = dx + 0.
        dx[1:, 1:] =  shifts_v 
        dx[0, 1:] = -shifts_v[0,:] # leftmost  column
        dx[:, 0] = dx[:,1] # bottom row
        dy[1:,1:] = shifts_h
        dy[1:,0] = -shifts_h[:,0] # bottom row
        dy[0, :] = dy[1,:] # leftmost column
        return dx,dy

    
    def EvalAreaChangeCorners(self, imax, z_end=None):
        """
        pixel area alterations computed through corner shifts
        """
        dx,dy = dx_dy(imax,zend)
        area_change = dx[1:,1:]-dx[:-1,1:]+dx[1:,:-1] - dx[:-1,:-1]
        area_change += dy[1:,1:]-dy[1:,:-1]+dy[:-1,1:]- dy[:-1,:-1]
        return -0.5*area_change
        
        
        area_change_h = shifts_h # dy top right corner 
        area_change_h[1:,:] = shift_h[1:,:] # dy top left corner
        area_change_h[1:,:] = shift_h[1:,:] # dy down right 
        
        area_change[0, :] -= shifts_v[0:,:]
        
        area_change[:, 0] -= shifts_h[:,0]
        area_change[1:,:] += shifts_v[:-1, :]
        area_change[:,1:] += shifts_h[:, :-1]
        return area_change
        
    

    def EvalAreaChangeSides(self, imax, z_end=None):
        """
        Same as EvalAreaChange, but twice as fast because symetries 
        are accounted for 
        """
        last_i = imax
        shifts_h = np.ndarray((last_i, last_i))
        shifts_v = np.ndarray((last_i, last_i))
        for i in range(last_i):
            for j in range(last_i):
                shifts_h[i,j] = self.average_shift_h(i,j,+1, z_end)
                shifts_v[i,j] = self.average_shift_v(i,j,+1, z_end)
        area_change = -shifts_h-shifts_v
        area_change[0, :] -= shifts_v[0,:]
        area_change[:, 0] -= shifts_h[:,0]
        area_change[1:,:] += shifts_v[:-1, :]
        area_change[:,1:] += shifts_h[:, :-1]
        return area_change

    

    def EvalAreaChangeSidesFast(self, imax, z_end=None, npair=11):
        """
        Same as EvalAreaChangeSides, but uses direct integration 
        """
        last_i = imax
        shifts_h = np.ndarray((last_i, last_i))
        shifts_v = np.ndarray((last_i, last_i))
        for i in range(last_i):
            for j in range(last_i):
                shifts_h[i,j] = self.Integ_Ey_fast(i,j, 1, z_end, npair=npair)
                shifts_v[i,j] = self.Integ_Ex_fast(i,j, 1, z_end, npair=npair)
        area_change = -shifts_h-shifts_v
        area_change[0, :] -= shifts_v[0,:]
        area_change[:, 0] -= shifts_h[:,0]
        area_change[1:,:] += shifts_v[:-1, :]
        area_change[:,1:] += shifts_h[:, :-1]
        return area_change


    
    def EvalAreaChange(self,i, j, z_end=None) :
        """
        i,j: integer offsets
        z_end : t by default, else lower values (to allow for red photons)
        """
        z_end = self.t if (z_end==None) else z_end
        return -(self.average_shift_h(i,j,-1, z_end)+self.average_shift_h(i,j,+1, z_end)+
        self.average_shift_v(i,j,-1, z_end)+self.average_shift_v(i,j,+1, z_end))


    
def hsc_ccd():
    dict = {'z_q': 5.58829214, 'zsh': 5.59954644,
            'zsv': 7.66267472, 'a': 2.85267152,
            'b': 5.23366811, 'thickness': 200.,
            'pixsize': 15.}
    return CcdGeom(**dict)

    
"""
"""
