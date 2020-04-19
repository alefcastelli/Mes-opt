# Analizing and Plotting results

from optimization_model_concrete import *

model.solutions.load_from(results)
fuel_In=[]
z=[]
Heat_gen=[]
Cold_gen=[]
El_gen=[]
Heat_us=[]
El_In=[]
Heat_diss=[]
el_grid=[]
El_tot=[]
s=[]
stor_lev=[]
stor_charge=[]
stor_disch=[]
delta_on=[]
delta_off=[]
z_design=[]
x_design=[]
gamma=[]
b=[]
Cinv=[]
beta=[]
psi=[]
PV_area=[]
El_gen_Res=[]
var_list=[z_design, z , delta_on, delta_off, b, s, x_design, PV_area, beta, psi, fuel_In, El_In, Heat_gen, Cold_gen, \
          El_gen, El_gen_Res, Heat_us, Heat_diss, el_grid, stor_lev, stor_charge, stor_disch, gamma, Cinv, El_tot]

i = 0
for v in model.component_objects(Var, active=True):
    print ("Variable",v)
    for index in v:
        print('  ', index, value(v[index]))
        var_list[i].append(value(v[index]))
    i=i+1


n_machines=len(model.Machines)
n_machines_heat=len(model.Machines_heat)
n_machines_cold=len(model.Machines_cold)
n_machines_el=len(model.Machines_el)
n_machines_elIn=len(model.Machines_elIn)
n_machines_diss=len(model.Machines_diss)

x_design=np.array(x_design).reshape(n_slots, n_machines)

z=np.array(z).reshape(n_slots*n_machines, T)
Heat_gen=np.array(Heat_gen).reshape(n_slots*n_machines_heat, T)
Heat_us=np.array(Heat_us).reshape(n_slots*n_machines_heat, T)
Cold_gen=np.array(Cold_gen).reshape(n_slots*n_machines_cold, T)
El_gen=np.array(El_gen).reshape(n_slots*n_machines_el, T)
El_In=np.array(El_In).reshape(n_slots*n_machines_elIn, T)
Heat_diss=np.array(Heat_diss).reshape(n_slots*n_machines_diss, T)

#Dictionary containing the results to be plotted
res={}
count=0
for i in model.Machines_heat:
    res["Heat_gen_{0}".format(i)]=0
    for j in range(n_slots):
        res["Heat_gen_{0}".format(i)] += Heat_gen[count+j, :]
    count+=n_slots
count=0

for i in model.Machines_cold:
    res["Cold_gen_{0}".format(i)]=0
    for j in range(n_slots):
        res["Cold_gen_{0}".format(i)] += Cold_gen[count+j, :]
    count+=n_slots

count=0
for i in model.Machines_el:
    res["El_gen_{0}".format(i)]=0
    for j in range(n_slots):
        res["El_gen_{0}".format(i)] += El_gen[count+j, :]
    count+=n_slots

res["El_gen_ICE"]=np.zeros(T)
res["Heat_gen_ICE"]=np.zeros(T)
res["Heat_gen_Boiler"]=np.zeros(T)
res["Heat_gen_HP"]=np.zeros(T)
res["Cold_gen_CC"]=np.zeros(T)
N_machines_per_type=3
for k in res.keys():
    for i in model.Machines_el:
        if k == "El_gen_{0}".format(i):
            res["El_gen_ICE".format(i)] += res[k]
    for j in range(1, N_machines_per_type+1):
        if k == "Heat_gen_ICE{0}".format(j):
            res["Heat_gen_ICE"] += res[k]
        if k == "Heat_gen_Boiler{0}".format(j):
            res["Heat_gen_Boiler"] += res[k]
        if k == "Heat_gen_HP{0}".format(j):
            res["Heat_gen_HP"] += res[k]
    for i in model.Machines_cold:
        if k == "Cold_gen_{0}".format(i):
            res["Cold_gen_CC"] += res[k]



El_consumed=- sum(El_In[:,])  #---> in teoria si potrebbe specificare consumata/dissipata da chi
Q_dissipated = - sum(Heat_diss[:,])

el_grid=np.array(el_grid)
s=np.array(s)
#el_purch=np.array(el_purch)
#el_sold=-np.array(el_sold)
#el_purch=el_grid*(1-s)
#el_sold =-el_grid*s

El_gen_PV=np.array(El_gen_Res)

stor_l=np.array(stor_lev)
Q_charge=-np.array(stor_charge)
Q_discharge=np.array(stor_disch)

'''
H_day=t
times_step=np.arange(H_day)
Fig_Q=plt.figure()
plt.xticks(times_step, list(times_step))
plt.ylabel("Heat [kWh]")
plt.title("Heat balance")
plt.bar(times_step, Q_discharge[0:H_day], bottom=res["Heat_gen_HP"][0:H_day]+res["Heat_gen_Boiler"][0:H_day] + res["Heat_gen_ICE"][0:H_day], label='TES discharge')
#plt.bar(times_step, res["Q_prod_ICE1"][0:H_day],  bottom=res["Q_prod_Boiler1"][0:H_day]+ res["Q_prod_Boiler2"][0:H_day] +res["Q_prod_Boiler3"][0:H_day]+ res["Q_prod_ICE2"][0:H_day] +res["Q_prod_ICE3"][0:H_day], label='Q prod ICE1')
#plt.bar(times_step, res["Q_prod_ICE2"][0:H_day],  bottom=res["Q_prod_Boiler1"][0:H_day]+ res["Q_prod_Boiler2"][0:H_day] +res["Q_prod_Boiler3"][0:H_day] +res["Q_prod_ICE3"][0:H_day], label='Q prod ICE2')
#plt.bar(times_step, res["Q_prod_ICE3"][0:H_day],  bottom=res["Q_prod_Boiler1"][0:H_day]+ res["Q_prod_Boiler2"][0:H_day] +res["Q_prod_Boiler3"][0:H_day], label='Q prod ICE3')
plt.bar(times_step, res["Heat_gen_ICE"][0:H_day],  bottom=res["Heat_gen_HP"][0:H_day]+res["Heat_gen_Boiler"][0:H_day], label='Q prod ICEs')
#plt.bar(times_step, res["Q_prod_Boiler1"][0:H_day], bottom=res["Q_prod_Boiler2"][0:H_day] +res["Q_prod_Boiler3"][0:H_day], label='Q prod Boiler1')
#plt.bar(times_step, res["Q_prod_Boiler2"][0:H_day], bottom=res["Q_prod_Boiler3"][0:H_day], label='Q prod Boiler2')
#plt.bar(times_step, res["Q_prod_Boiler3"][0:H_day], label='Q prod Boiler3')
plt.bar(times_step, res["Heat_gen_Boiler"][0:H_day], bottom=res["Heat_gen_HP"][0:H_day], label='Heat gen Boilers')
plt.bar(times_step, res["Heat_gen_HP"][0:H_day], label='Heat gen HP')
plt.bar(times_step, Q_charge[0:H_day], label='TES charge')
plt.bar(times_step, Q_dissipated[0:H_day], bottom=Q_charge[0:H_day], label='Heat diss')


plt.plot(times_step, np.array(Heat_demand[0:H_day]), 'k--', label='Heat demand')
plt.plot(times_step, stor_l[0:H_day], 'b--', label='Storage level')
plt.legend()

Fig_E=plt.figure()
plt.xticks(times_step, list(times_step))
plt.ylabel("Electric Energy [kWh]")
plt.title("Electricity balance")
plt.bar(times_step, el_purch[0:H_day], bottom=el_prod_PV[0:H_day]+res["El_gen_ICE"][0:H_day], label='el purch')
#plt.bar(times_step, res["El_prod_ICE1"][0:H_day], bottom=el_prod_PV[0:H_day]+ res["El_prod_ICE2"][0:H_day]+res["El_prod_ICE3"][0:H_day], label='el prod ICE1')
#plt.bar(times_step, res["El_prod_ICE2"][0:H_day], bottom=el_prod_PV[0:H_day]+res["El_prod_ICE3"][0:H_day], label='el prod ICE2')
#plt.bar(times_step, res["El_prod_ICE3"][0:H_day], bottom=el_prod_PV[0:H_day], label='el prod ICE3')
plt.bar(times_step, res["El_gen_ICE"][0:H_day], bottom=el_prod_PV[0:H_day], label='El gen ICEs')
plt.bar(times_step, el_prod_PV[0:H_day], label='PV output')
plt.bar(times_step, el_sold[0:H_day], label='el sold')
plt.bar(times_step, El_consumed[0:H_day], label='el consumed')

plt.plot(times_step, np.array(EE_demand[0:H_day]), 'r--', label='El demand')
plt.legend()
plt.show()

'''

### PLOTS with Pandas ##
plt.figure()
df=pd.DataFrame.from_dict(res)
df_Heat=df[["Heat_gen_ICE", "Heat_gen_Boiler", "Heat_gen_HP"]].copy()
df_Heat["Heat_diss"]=Q_dissipated
df_Heat["Stor_charge"]=Q_charge
df_Heat["Stor_discharge"]=Q_discharge
df["Heat Demand"] = Heat_demand
df["Storage Level"]=stor_lev
ax1=df["Heat Demand"].plot(kind='line', color='r', linestyle='--', label='Heat demand', legend=True)
df["Storage Level"].plot(kind='line', ax=ax1, color='k', linestyle='--', label='Storage level', legend=True)
df_Heat.plot(kind='bar', stacked=True, ax=ax1)
plt.xlabel("Timestep [h]")
plt.ylabel("Energy[kWh]")
plt.title("Heat Production")

plt.figure()
df_El=pd.DataFrame(res["El_gen_ICE"], columns=["El gen ICE"])
df_El["El gen PV"]=El_gen_PV
df_El["El_consumed"]=El_consumed
#df_El["El sold"]=el_sold
#df_El["El purch"]=el_purch
df_El["El grid"]=-el_grid
df["El Demand"]=EE_demand
ax2=df["El Demand"].plot(kind='line', color='g', linestyle='--', label='El demand', legend=True)
df_El.plot(kind='bar', stacked=True, ax=ax2, legend=True)
plt.xlabel("Timestep [h]")
plt.ylabel("Energy[kWh]")
plt.title("Electricity Production")

plt.figure()
df_Cold=pd.DataFrame(res["Cold_gen_CC"], columns=["Cold gen CC"])
df["Cold Demand"]=Cold_demand
ax3=df["Cold Demand"].plot(kind='line', color='b', linestyle='--', label='Cold demand', legend=True)
df_Cold["Cold gen CC"].plot(kind='bar', stacked=True, ax=ax3, legend=True)
plt.xlabel("Timestep [h]")
plt.ylabel("Energy[kWh]")
plt.title("Cold Production")
