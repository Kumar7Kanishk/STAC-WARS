'''
Here I will Analyze the given data and tell which planet is more habitable.
'''


import pandas as pd
import math as m


'''It will provide the time period of planets around STAC-1, STAC-2, STAC-3'''

stac1 = pd.read_csv('stac1_brightness.csv')
stac2 = pd.read_csv('stac2_brightness.csv')
stac3 = pd.read_csv('stac3_brightness.csv')


filt1 = (stac1['Rel_brightness'] != 1)
diffr1 = stac1.loc[filt1, 'Time_days'].diff().dropna().rename("STAC-1")

filt2 = (stac2['Rel_brightness'] != 1)
diffr2 = stac2.loc[filt2, 'Time_days'].diff().dropna().rename("STAC-2")

filt3 = (stac3['Rel_brightness'] != 1)
diffr3 = stac3.loc[filt3, 'Time_days'].diff().dropna().rename("STAC-3")

diffr1 = diffr1.reset_index(drop=True)
diffr2 = diffr2.reset_index(drop=True)
diffr3 = diffr3.reset_index(drop=True)

combined = pd.concat([diffr1, diffr2, diffr3], axis=1)

combined.columns = ["STAC-1", "STAC-2", "STAC-3"]

print(combined)
print()
print("Since planets around STAC-1 and STAC-2 has minimum error in their time-period, So we consider planets 1 and 2 as our transiting exoplanet.")
print()
strr = "STAC-1 and STAC-2 show consistent transit periods (4-5 days and 80 days respectively), indicating periodic planetary transits. STAC-3 exhibits irregular intervals (40, 40, 40, 30 days), making the evidence for a transiting planet less reliable."
print(strr)
print()
avg_stac1 = diffr1.mean()
avg_stac2 = diffr2.mean()

orb_years1 = avg_stac1/365.25
orb_years2 = avg_stac2/365.25

'''
From the output we can say that out of these 3 , exoplanets moving around STAC-1 
and STAC-2 can be considered as an transiting exoplanet with a time period 
error in observation of STAC-1 brightness level of 1 day, in STAC-3 of 
10 days and Zero error in STAC-2
'''
#==========================================================================================#

'''This will give transit-depth of exoplanets found.
Transit-depth(δ) = (Fnormal - Ftransit) / Fnormal
'''
stac1 = pd.read_csv('stac1_brightness.csv')
stac2 = pd.read_csv('stac2_brightness.csv')
stellr_Obs = pd.read_csv('stellar_observations_data.csv')

uniq_bright1 = stac1['Rel_brightness'].unique()
del1 = (uniq_bright1[0]-uniq_bright1[1])/uniq_bright1[0]

uniq_bright2 = stac2['Rel_brightness'].unique()
del2 = (uniq_bright2[0]-uniq_bright2[1])/uniq_bright2[0]

#==========================================================================================#

'''
Here Radius of transiting-Exoplanet found using formula:
δ ≈ (Rs / Rp)2
'''
rs1 = float(stellr_Obs['STAC-1'].iloc[1].replace("R☉", "").strip())
rs2 = float(stellr_Obs['STAC-2'].iloc[1].replace("R☉", "").strip())

r_sun = 696000
r_earth = 6371
rp1 = rs1*m.sqrt(del1)*r_sun
rp2 = rs2*m.sqrt(del2)*r_sun

rp1_earth = rp1/r_earth
rp2_earth = rp2/r_earth

#==========================================================================================#

'''
Planets are being classified
'''
def classify_planet(Rp_earth):
    if Rp_earth < 1.25:
        return "Earth-sized"
    elif Rp_earth < 2:
        return "Super-Earth"
    elif Rp_earth < 6:
        return "Neptune-sized"
    else:
        return "Jupiter-sized"

#==========================================================================================#

'''
We found out Orbital distance using formula :
P^2 = a^3 / M 
'''
ms1 = float(stellr_Obs['STAC-1'].iloc[0].replace("M☉", "").strip())
ms2 = float(stellr_Obs['STAC-2'].iloc[0].replace("M☉", "").strip())

def orb_dist(M,P):
    a = (M * P**2) ** (1/3)
    return a

AU = 149.6 * (10**6)

#==========================================================================================#

'''
Here Habitable zone distance using formula :
dHZ ≈ √(L / L☉) AU
'''
lum_sun = 1
lum_1_sun = float(stellr_Obs['STAC-1'].iloc[2].replace("L☉", "").strip())
lum_2_sun = float(stellr_Obs['STAC-2'].iloc[2].replace("L☉", "").strip())

def dhz(lum_x_sun,lum_sun):
    dhzz = m.sqrt(lum_x_sun/lum_sun)
    return dhzz

#==========================================================================================#

'''
Report of the two transiting-exoplanets using the found parameters
'''
report = pd.DataFrame({
    "Property": [
        "Average Period (days)",
        "Transit Depth",
        "Radius (km)",
        "Radius (Earth Radii)",
        "Planet Type",
        "Orbital Distance (AU)",
        "Orbital Distance (km)",
        "Habitable Zone Distance (AU)"
    ],
    "Planet-1": [
        round(avg_stac1, 2),
        round(del1, 4),
        round(rp1, 2),
        round(rp1_earth, 2),
        classify_planet(rp1_earth),
        round(orb_dist(ms1, orb_years1), 4),
        round(orb_dist(ms1, orb_years1) * AU, 2),
        dhz(lum_1_sun,lum_sun)
    ],
    "Planet-2": [
        round(avg_stac2, 2),
        round(del2, 4),
        round(rp2, 2),
        round(rp2_earth, 2),
        classify_planet(rp2_earth),
        round(orb_dist(ms2, orb_years2), 4),
        round(orb_dist(ms2, orb_years2) * AU, 2),
        round(dhz(lum_2_sun,lum_sun), 2)
    ]
})

report.set_index("Property", inplace=True)
print("\n" + "="*70)
print("                       FINAL EXOPLANET REPORT")
print("="*70)

print(report.to_markdown())

print("="*70)
print()

#==========================================================================================#

'''
Conclusion
'''

strr = '''Planet-2 is the most promising candidate for further habitability studies.
Evidence supporting this choice:

1. Planet-2 orbits at approximately 0.337 AU, which is much closer to the estimated habitable-zone distance of 0.632 AU than Planet-1, which orbits at only 0.051 AU.
2. Planet-2 receives less stellar radiation because it is farther from its host star. Planet-1 is extremely close to its star and is therefore likely too hot to support liquid water.
3. Planet-2 has a smaller radius than Planet-1, making it a more favorable candidate for future investigations of planetary composition and potential habitability'''

print(strr)
