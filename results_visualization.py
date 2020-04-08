# Analizing and Plotting results

from optimization_model_concrete import *
import pandas as pd

model.solutions.load_from(results)

F_In=[]
z=[]
Q_prod=[]
El_prod=[]
Q_us=[]
El_us=[]
El_cons=[]
Q_diss=[]
el_purch=[]
el_sold=[]
stor_lev=[]
stor_charge=[]
stor_disch=[]
power_in=[]
power_out=[]
delta_on=[]
delta_off=[]
z_design=[]
PV_area=[]
el_prod_PV=[]
var_list=[F_In ,z , delta_on, delta_off, Q_prod, El_prod, Q_us, El_us, El_cons, Q_diss, \
          el_purch, el_sold, stor_lev, stor_charge, stor_disch, power_in, power_out, z_design, PV_area, el_prod_PV]

i = 0
for v in model.component_objects(Var, active=True):
    print ("Variable",v)
    for index in v:
        print('  ', index, value(v[index]))
        var_list[i].append(value(v[index]))
    i=i+1


n_machines=len(model.Machines)
n_machines_el=len(model.Machines_el)
n_machines_int=len(model.Machines_int)

z=np.array(z).reshape(n_slots*n_machines, T)
Q_prod=np.array(Q_prod).reshape(n_slots*n_machines, T)
Q_us=np.array(Q_us).reshape(n_slots*n_machines, T)
Q_prod=np.array(Q_prod).reshape(n_slots*n_machines, T)

El_prod=np.array(El_prod).reshape(n_slots*n_machines_el, T)
El_us=np.array(El_us).reshape(n_slots*n_machines_el, T)
El_cons=np.array(El_cons).reshape(n_slots*n_machines_int, T)
Q_diss=np.array(Q_diss).reshape(n_slots*n_machines_el, T)

#Dictionary containing the results to be plotted
res={}
count=0
for i in model.Machines_heat:
    res["Q_prod_{0}".format(i)]=0
    for j in range(n_slots):
        res["Q_prod_{0}".format(i)] += Q_prod[count+j, :]
    count+=n_slots

count=0
for i in model.Machines_el:
    res["El_prod_{0}".format(i)]=0
    for j in range(n_slots):
        res["El_prod_{0}".format(i)] += El_prod[count+j, :]
    count+=n_slots

res["El_prod_ICE"]=np.zeros(T)
res["Q_prod_ICE"]=np.zeros(T)
res["Q_prod_Boiler"]=np.zeros(T)
res["Q_prod_HP"]=np.zeros(T)
for k in res.keys():
    for i in model.Machines_el:
        if k == "El_prod_{0}".format(i):
            res["El_prod_ICE"] += res[k]
    for j in range(1, len(model.Machines_heat)+1):
        if k == "Q_prod_ICE{0}".format(j):
            res["Q_prod_ICE"] += res[k]
        if k == "Q_prod_Boiler{0}".format(j):
            res["Q_prod_Boiler"] += res[k]
        if k == "Q_prod_HP{0}".format(j):
            res["Q_prod_HP"] += res[k]



El_consumed=- sum(El_cons[:,])
Q_dissipated = - sum(Q_diss[:,])

el_purch=np.array(el_purch)
el_sold=-np.array(el_sold)

el_prod_PV=np.array(el_prod_PV)

stor_l=np.array(stor_lev)
Q_charge=-np.array(stor_charge)
Q_discharge=np.array(stor_disch)

H_day=t
times_step=np.arange(H_day)
Fig_Q=plt.figure()
plt.xticks(times_step, list(times_step))
plt.ylabel("Heat [kWh]")
plt.title("Q balance")
plt.bar(times_step, Q_discharge[0:H_day], bottom=res["Q_prod_HP"][0:H_day]+res["Q_prod_Boiler"][0:H_day] + res["Q_prod_ICE"][0:H_day], label='TES discharge')
#plt.bar(times_step, res["Q_prod_ICE1"][0:H_day],  bottom=res["Q_prod_Boiler1"][0:H_day]+ res["Q_prod_Boiler2"][0:H_day] +res["Q_prod_Boiler3"][0:H_day]+ res["Q_prod_ICE2"][0:H_day] +res["Q_prod_ICE3"][0:H_day], label='Q prod ICE1')
#plt.bar(times_step, res["Q_prod_ICE2"][0:H_day],  bottom=res["Q_prod_Boiler1"][0:H_day]+ res["Q_prod_Boiler2"][0:H_day] +res["Q_prod_Boiler3"][0:H_day] +res["Q_prod_ICE3"][0:H_day], label='Q prod ICE2')
#plt.bar(times_step, res["Q_prod_ICE3"][0:H_day],  bottom=res["Q_prod_Boiler1"][0:H_day]+ res["Q_prod_Boiler2"][0:H_day] +res["Q_prod_Boiler3"][0:H_day], label='Q prod ICE3')
plt.bar(times_step, res["Q_prod_ICE"][0:H_day],  bottom=res["Q_prod_HP"][0:H_day]+res["Q_prod_Boiler"][0:H_day], label='Q prod ICEs')
#plt.bar(times_step, res["Q_prod_Boiler1"][0:H_day], bottom=res["Q_prod_Boiler2"][0:H_day] +res["Q_prod_Boiler3"][0:H_day], label='Q prod Boiler1')
#plt.bar(times_step, res["Q_prod_Boiler2"][0:H_day], bottom=res["Q_prod_Boiler3"][0:H_day], label='Q prod Boiler2')
#plt.bar(times_step, res["Q_prod_Boiler3"][0:H_day], label='Q prod Boiler3')
plt.bar(times_step, res["Q_prod_Boiler"][0:H_day], bottom=res["Q_prod_HP"][0:H_day], label='Q prod Boilers')
plt.bar(times_step, res["Q_prod_HP"][0:H_day], label='Q prod HP')
plt.bar(times_step, Q_charge[0:H_day], label='TES charge')
plt.bar(times_step, Q_dissipated[0:H_day], bottom=Q_charge[0:H_day], label='Q diss')


plt.plot(times_step, np.array(Heat_demand[0:H_day]), 'k--', label='Q demand')
plt.plot(times_step, stor_l[0:H_day], 'b--', label='Storage level')
plt.legend()

Fig_E=plt.figure()
plt.xticks(times_step, list(times_step))
plt.ylabel("Electric Energy [kWh]")
plt.title("Electricity balance")
plt.bar(times_step, el_purch[0:H_day], bottom=el_prod_PV[0:H_day]+res["El_prod_ICE"][0:H_day], label='el purch')
#plt.bar(times_step, res["El_prod_ICE1"][0:H_day], bottom=el_prod_PV[0:H_day]+ res["El_prod_ICE2"][0:H_day]+res["El_prod_ICE3"][0:H_day], label='el prod ICE1')
#plt.bar(times_step, res["El_prod_ICE2"][0:H_day], bottom=el_prod_PV[0:H_day]+res["El_prod_ICE3"][0:H_day], label='el prod ICE2')
#plt.bar(times_step, res["El_prod_ICE3"][0:H_day], bottom=el_prod_PV[0:H_day], label='el prod ICE3')
plt.bar(times_step, res["El_prod_ICE"][0:H_day], bottom=el_prod_PV[0:H_day], label='el prod ICEs')
plt.bar(times_step, el_prod_PV[0:H_day], label='PV output')
plt.bar(times_step, el_sold[0:H_day], label='el sold')
plt.bar(times_step, El_consumed[0:H_day], label='el consumed')

plt.plot(times_step, np.array(EE_demand[0:H_day]), 'r--', label='El demand')
plt.legend()
plt.show()



### PLOTS with Pandas ##

df=pd.DataFrame.from_dict(res)
df_Heat=df[["Q_prod_ICE", "Q_prod_Boiler", "Q_prod_HP"]]
df_Heat["Q_diss"]=Q_dissipated
df_Heat["stor_charge"]=Q_charge
df_Heat["stor_discharge"]=Q_discharge
df["Heat Demand"] = Heat_demand
ax1 = df_Heat.plot(kind='bar', stacked=True, grid=True)
df["Heat Demand"].plot(kind='line', ax=ax1)
plt.xlabel("Timestep [h]")
plt.ylabel("Energy[kWh]")
plt.title("Heat Production")

df_El=pd.DataFrame(res["El_prod_ICE"])
df_El["El prod PV"]=el_prod_PV
df_El["El_consumed"]=El_consumed
df_El["El sold"]=el_sold
df_El["El purch"]=el_purch
df["El Demand"]=EE_demand
ax2 = df_El.plot(kind='bar', stacked=True, grid=True)
df["El Demand"].plot(kind='line', color='k', ax=ax2)
plt.xlabel("Timestep [h]")
plt.ylabel("Energy[kWh]")
plt.title("Electricity Production")
