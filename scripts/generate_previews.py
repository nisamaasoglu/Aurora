"""
AURORA - README asset generator (volumetric edition)
- Renders preview stills and the motion GIF by running the volumetric
  raymarching math from public/index.html faithfully in numpy (volumetric.py).
- Runs the DSP analysis pipeline on a synthetic piano signal to produce a real
  numeric-output image (console_output.png).

Note: previews are real renders computed from the shader math, not live browser
screenshots (same algorithm, same result). Rendering is CPU-side and slow
(~1 min per still) - the live app runs the same math on the GPU at 60 fps.

Usage:
    python scripts/generate_previews.py still calm|happy|angry
    python scripts/generate_previews.py interface
    python scripts/generate_previews.py console
    python scripts/generate_previews.py gif        # renders 20 frames; slow
"""
import os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from volumetric import render

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "images")
os.makedirs(OUT, exist_ok=True)
W, H = 720, 450

STATES = {
    "calm":  dict(emo=0.0, pitch=0.15, rms=0.10),
    "happy": dict(emo=0.5, pitch=0.45, rms=0.22),
    "angry": dict(emo=1.0, pitch=0.75, rms=0.34),
}

def still(name):
    s = STATES[name]
    img = render(W, H, t_time=8.0, rms=s["rms"], pitch=s["pitch"],
                 emo_from=s["emo"], emo_to=s["emo"], blend=1.0, seed=0.421)
    im = Image.fromarray((img*255).astype("uint8"))
    im.save(os.path.join(OUT, f"preview_{name}.png"))
    return im

def _font(mono, size):
    import matplotlib
    base = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    name = "DejaVuSansMono.ttf" if mono else "DejaVuSans.ttf"
    try: return ImageFont.truetype(os.path.join(base, name), size)
    except Exception: return ImageFont.load_default()

def hud(img, d):
    im = img.copy(); dr = ImageDraw.Draw(im, "RGBA")
    acc = {"calm":(63,160,255),"happy":(31,214,138),"angry":(255,90,42)}[d["emotion"]]
    dr.text((24,20), "A U R O R A", font=_font(0,22), fill=(255,255,255,235))
    dr.text((26,50), "LIVE SOUND · GENERATIVE LIGHT", font=_font(0,10), fill=(255,255,255,140))
    dr.ellipse((W-130,24,W-123,31), fill=(53,208,127,255))
    dr.text((W-114,21), "LIVE", font=_font(0,11), fill=(255,255,255,150))
    px,py = 24, H-168; pw = 218
    dr.rounded_rectangle((px,py,px+pw,py+144), 12, fill=(10,12,20,150), outline=(255,255,255,40))
    rows = [("NOTE", d["note"]), ("PITCH", f"{d['pitch']} Hz"),
            ("EMOTION", d["emotion"]), ("ENERGY", f"{d['rms']:.3f}"), ("FPS", "60")]
    y = py+13
    for lab,val in rows:
        dr.text((px+15,y+2), lab, font=_font(0,10), fill=(255,255,255,140))
        vcol = acc+(255,) if lab=="EMOTION" else (255,255,255,235)
        w = dr.textlength(val, font=_font(1,13))
        dr.text((px+pw-15-w,y), val, font=_font(1,13), fill=vcol)
        y += 24 if lab!="ENERGY" else 17
        if lab=="ENERGY":
            dr.rounded_rectangle((px+15,y,px+pw-15,y+3),2, fill=(255,255,255,30))
            fw_=int((pw-30)*min(d["rms"]*4,1))
            dr.rounded_rectangle((px+15,y,px+15+fw_,y+3),2, fill=acc+(255,))
            y += 18
    dr.text((W-125,H-58), f"#{d['seed']:05d}", font=_font(1,21), fill=acc+(255,))
    dr.text((W-160,H-32), "THIS MOMENT CAN'T BE REPEATED", font=_font(0,8), fill=(255,255,255,140))
    return im

def interface():
    base_path = os.path.join(OUT, "preview_happy.png")
    base = Image.open(base_path) if os.path.exists(base_path) else still("happy")
    hud(base, dict(note="D", pitch=587, emotion="happy", rms=0.22, seed=42137)) \
        .save(os.path.join(OUT, "interface.png"))

# ---- DSP console demo (numpy pipeline mirroring the engine's formulas) ----
def synth_note(freq, sr=44100, n=2048, amp=0.2, noise=0.03):
    t = np.arange(n)/sr
    x = amp*np.sin(2*np.pi*freq*t) + 0.4*amp*np.sin(2*np.pi*freq*2*t) \
        + 0.2*amp*np.sin(2*np.pi*freq*3*t)
    return (x + np.random.normal(0, noise, n)).astype(np.float32)

def features(x, sr=44100):
    rms = float(np.sqrt(np.mean(x**2)))
    win = np.hanning(len(x)); X = np.abs(np.fft.rfft(x*win))
    fr = np.fft.rfftfreq(len(x), 1/sr)
    centroid = float(np.sum(fr*X)/(np.sum(X)+1e-9))
    csum = np.cumsum(X); rolloff = float(fr[np.searchsorted(csum, 0.85*csum[-1])])
    ac = np.correlate(x, x, "full")[len(x)-1:]
    lo, hi = int(sr/2093), int(sr/65)
    lag = lo + int(np.argmax(ac[lo:hi])); pitch = sr/lag if lag else 0.0
    names=["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    midi = int(round(69+12*np.log2(pitch/440))) if pitch>0 else 69
    return rms, centroid, rolloff, pitch, names[midi % 12]

def emo_rule(rms, centroid, rolloff):
    cn=min(centroid/5000,1); rn=min(rolloff/15000,1); en=min(rms*100,1)
    br=(cn+rn)/2
    if en<0.15: return "calm" if br<0.3 else ("neutral" if br<0.6 else "sad")
    if en<0.5:  return "neutral" if br<0.3 else ("happy" if br<0.6 else "excited")
    return "angry" if br<0.4 else ("excited" if br<0.7 else "happy")

def console():
    seq = [("C4",261.6,0.22,0.03),("E4",329.6,0.20,0.03),("G4",392.0,0.24,0.04),
           ("C5",523.3,0.31,0.05),("A4",440.0,0.12,0.03),("D5",587.3,0.28,0.05),
           ("F5",698.5,0.34,0.06),("B4",493.9,0.09,0.02)]
    lines=[]
    for _,freq,amp,ns in seq:
        x=synth_note(freq, amp=amp, noise=ns)
        rms,cen,rol,pit,note=features(x)
        e=emo_rule(rms,cen,rol)
        seed=abs(hash(f"{cen:.2f}{rol:.2f}{pit:.2f}"))%100000
        lines.append((f"NOTE: {note:<2} | PITCH: {pit:6.1f} Hz | CENTROID: {cen:6.0f} | "
                      f"RMS: {rms:.3f} | EMOTION: {e:<8} | ID: #{seed:05d}", e))
    fh=_font(1,17); pad=18; lh=26
    im=Image.new("RGB",(980, pad*2+lh*len(lines)),(12,14,18))
    dr=ImageDraw.Draw(im)
    cmap={"calm":(90,160,255),"happy":(240,205,70),"excited":(200,120,255),
          "sad":(90,205,220),"angry":(255,95,60),"neutral":(210,210,210)}
    for i,(ln,e) in enumerate(lines):
        dr.text((pad, pad+i*lh), ln, font=fh, fill=cmap.get(e,(220,220,220)))
    im.save(os.path.join(OUT,"console_output.png"))

def gif():
    frames=[]
    N=20
    for i in range(N):
        p=i/(N-1)
        img = render(360, 225, t_time=4.0+i*0.5,
                     rms=0.14+0.12*abs(np.sin(p*np.pi*3)), pitch=0.35,
                     emo_from=0.0, emo_to=1.0, blend=p, seed=0.421,
                     steps=36, octaves=3)
        frames.append(Image.fromarray((img*255).astype("uint8")))
        print("frame", i, flush=True)
    seq=frames+frames[-2:0:-1]
    seq[0].save(os.path.join(OUT,"aurora_live.gif"), save_all=True,
                append_images=seq[1:], duration=110, loop=0, optimize=True)

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv)>1 else "console"
    if cmd == "still": still(sys.argv[2]); print("ok")
    elif cmd == "interface": interface(); print("ok")
    elif cmd == "console": console(); print("ok")
    elif cmd == "gif": gif(); print("ok")
