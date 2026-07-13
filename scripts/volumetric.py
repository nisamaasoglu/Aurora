"""Numpy port of the volumetric raymarching shader (public/index.html).
Used to render README stills and the motion GIF - same math, same result."""
import numpy as np

def _hash3(x, y, z):
    px = np.modf(x*0.3183099 + 0.1)[0]*17.0
    py = np.modf(y*0.3183099 + 0.2)[0]*17.0
    pz = np.modf(z*0.3183099 + 0.3)[0]*17.0
    return np.modf(px*py*pz*(px+py+pz))[0]

def _noise3(x, y, z):
    ix, iy, iz = np.floor(x), np.floor(y), np.floor(z)
    fx, fy, fz = x-ix, y-iy, z-iz
    fx = fx*fx*(3-2*fx); fy = fy*fy*(3-2*fy); fz = fz*fz*(3-2*fz)
    n000=_hash3(ix,iy,iz);     n100=_hash3(ix+1,iy,iz)
    n010=_hash3(ix,iy+1,iz);   n110=_hash3(ix+1,iy+1,iz)
    n001=_hash3(ix,iy,iz+1);   n101=_hash3(ix+1,iy,iz+1)
    n011=_hash3(ix,iy+1,iz+1); n111=_hash3(ix+1,iy+1,iz+1)
    a = n000*(1-fx)+n100*fx; b = n010*(1-fx)+n110*fx
    c = n001*(1-fx)+n101*fx; d = n011*(1-fx)+n111*fx
    return (a*(1-fy)+b*fy)*(1-fz) + (c*(1-fy)+d*fy)*fz

def _fbm3(x, y, z, octaves=4):
    v = np.zeros_like(x); a = 0.5
    for _ in range(octaves):
        v = v + a*_noise3(x, y, z)
        x = x*2.03+11.0; y = y*2.03+11.0; z = z*2.03+11.0
        a *= 0.5
    return v

def _fbm3lo(x, y, z):
    return 0.5*_noise3(x,y,z)+0.25*_noise3(x*2.03+11,y*2.03+11,z*2.03+11)+0.25

def _smoothstep(a, b, x):
    t = np.clip((x-a)/(b-a), 0, 1); return t*t*(3-2*t)

def palette(e):
    calm=np.array([.05,.25,.85]); happy=np.array([0,.85,.55]); angry=np.array([1,.28,.05])
    return (calm+(happy-calm)*(e*2)) if e<0.5 else (happy+(angry-happy)*((e-0.5)*2))

def render(W, H, t_time, rms, pitch, emo_from, emo_to, blend, seed,
           steps=44, octaves=4):
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    uvx = (xs*2.0 - W)/H
    uvy = -(ys*2.0 - H)/H          # flip: image y grows downward
    energy = min(rms*3.0, 1.5)

    ang = t_time*0.06
    ro = np.array([2.7*np.sin(ang), 0.35*np.sin(t_time*0.11), -2.7*np.cos(ang)])
    fw = -ro/np.linalg.norm(ro)
    rt = np.cross([0,1,0], fw); rt = rt/np.linalg.norm(rt)
    up = np.cross(fw, rt)
    rdx = uvx*rt[0] + uvy*up[0] + 1.5*fw[0]
    rdy = uvx*rt[1] + uvy*up[1] + 1.5*fw[1]
    rdz = uvx*rt[2] + uvy*up[2] + 1.5*fw[2]
    n = np.sqrt(rdx**2+rdy**2+rdz**2); rdx/=n; rdy/=n; rdz/=n

    flow_y = t_time*(0.10+0.35*energy)
    so = np.array([seed*37.0, seed*11.0, seed*23.0])
    bF, bT = palette(emo_from), palette(emo_to)
    accent = np.array([.55,.3,.95])+(np.array([1,.85,.4])-np.array([.55,.3,.95]))*pitch
    L = np.array([0.5,0.8,-0.3]); L = L/np.linalg.norm(L)

    col = np.zeros((H, W, 3)); trans = np.ones((H, W))
    t = 0.7
    for _ in range(steps):
        px = ro[0]+rdx*t; py = ro[1]+rdy*t; pz = ro[2]+rdz*t
        qx = px*0.85+so[0]; qy = py*0.85+flow_y+so[1]; qz = pz*0.85+so[2]
        w = _fbm3(qx*0.55+2.0, qy*0.55+2.0, qz*0.55+2.0, octaves)
        d = _fbm3(qx+2.0*w, qy+2.0*w, qz+2.0*w, octaves)
        shape = np.exp(-0.34*(px*px+py*py+pz*pz))
        dens = _smoothstep(0.42, 0.80, d)*shape

        active = dens > 0.004
        if active.any():
            dl = _fbm3(qx+L[0]*0.4+2.0*w, qy+L[1]*0.4+2.0*w, qz+L[2]*0.4+2.0*w, octaves)
            shade = np.clip((d-dl)*2.2+0.55, 0, 1)
            seep = _fbm3lo(qx*0.9+7.7, qy*0.9+3.3, qz*0.9+1.9)
            th = 1.05+(-0.05-1.05)*blend
            mk = _smoothstep(th-0.2, th+0.2, seep)[...,None]
            base = bF*(1-mk)+bT*mk
            base = base+(accent-base)*(0.22*pitch)
            c = base*(0.22+0.95*shade[...,None]) + accent*(shade[...,None]**4)*(0.5+energy)
            a = dens*(0.55+0.5*energy)*0.17*active
            col += c*(a*trans)[...,None]
            trans *= 1.0-a
        t += 0.085

    col *= 1.15+0.9*energy
    vig = _smoothstep(1.7, 0.3, np.sqrt(uvx**2+uvy**2))
    col *= (0.5+0.5*vig)[...,None]
    col += (np.random.rand(H,W,1)-0.5)*0.02
    return np.clip(col, 0, 1)
