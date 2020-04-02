# Analizing and Plotting results

from optimization_model_concrete import *

model.solutions.load_from(results)

F_In=[]
z=[]
Q_prod=[]
El_prod=[]
Q_us=[]
El_us=[]
Q_diss=[]
el_purch=[]
el_sold=[]
stor_lev=[]
power_in=[]
power_out=[]
delta_on=[]
delta_off=[]
z_design=[]
var_list=[F_In ,z , delta_on, delta_off, Q_prod, El_prod, Q_us, El_us, Q_diss, \
          el_purch, el_sold, stor_lev, power_in, power_out, z_design]

i = 0
for v in model.component_objects(Var, active=True):
    print ("Variable",v)
    for index in v:
        print('  ', index, value(v[index]))
        var_list[i].append(value(v[index]))
    i=i+1


n_machines=len(model.Machines)
n_machines_el=len(model.Machines_el)

z=np.array(z).reshape(n_slots*n_machines, T)
Q_prod=np.array(Q_prod).reshape(n_slots*n_machines, T)
Q_us=np.array(Q_us).reshape(n_slots*n_machines, T)
Q_prod=np.array(Q_prod).reshape(n_slots*n_machines, T)

El_prod=np.array(El_prod).reshape(n_slots*n_machines_el, T)
El_us=np.array(El_us).reshape(n_slots*n_machines_el, T)
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

Q_diss = - sum(Q_diss[:,])

el_purch=np.array(el_purch)
el_sold=-np.array(el_sold)

Q_charge=[0]
Q_discharge=[0]

for i in range(1, len(stor_lev)):
    diff=stor_lev[i]-stor_lev[i-1]
    if diff >=0:
        Q_charge.append(diff)
        Q_discharge.append(0)
    else:
        Q_discharge.append(diff)
        Q_charge.append(0)

stor_l=np.array(stor_lev)
Q_charge=-np.array(Q_charge)
Q_discharge=-np.array(Q_discharge)


H_day=24
times_step=np.arange(H_day)
Fig_Q=plt.figure()
plt.xticks(times_step, list(times_step))
plt.ylabel("Heat [kWh]")
plt.title("Q balance")
plt.bar(times_step, Q_discharge[0:H_day], bottom=res["Q_prod_Boiler1"][0:H_day]+ res["Q_prod_Boiler2"][0:H_day] + res["Q_prod_Boiler3"][0:H_day] + res["Q_prod_ICE1"][0:H_day] +res["Q_prod_ICE2"][0:H_day] + res["Q_prod_ICE3"][0:H_day], label='TES1 discharge')
plt.bar(times_step, res["Q_prod_ICE1"][0:H_day],  bottom=res["Q_prod_Boiler1"][0:H_day]+ res["Q_prod_Boiler2"][0:H_day] +res["Q_prod_Boiler3"][0:H_day]+ res["Q_prod_ICE2"][0:H_day] +res["Q_prod_ICE3"][0:H_day], label='Q prod ICE1')
plt.bar(times_step, res["Q_prod_ICE2"][0:H_day],  bottom=res["Q_prod_Boiler1"][0:H_day]+ res["Q_prod_Boiler2"][0:H_day] +res["Q_prod_Boiler3"][0:H_day] +res["Q_prod_ICE3"][0:H_day], label='Q prod ICE2')
plt.bar(times_step, res["Q_prod_ICE3"][0:H_day],  bottom=res["Q_prod_Boiler1"][0:H_day]+ res["Q_prod_Boiler2"][0:H_day] +res["Q_prod_Boiler3"][0:H_day], label='Q prod ICE3')
plt.bar(times_step, res["Q_prod_Boiler1"][0:H_day], bottom=res["Q_prod_Boiler2"][0:H_day] +res["Q_prod_Boiler3"][0:H_day], label='Q prod Boiler1')
plt.bar(times_step, res["Q_prod_Boiler2"][0:H_day], bottom=res["Q_prod_Boiler3"][0:H_day], label='Q prod Boiler2')
plt.bar(times_step, res["Q_prod_Boiler3"][0:H_day], label='Q prod Boiler3')
plt.bar(times_step, Q_charge[0:H_day], label='TES1 charge')
plt.bar(times_step, Q_diss[0:H_day], bottom=Q_charge[0:H_day], label='Q diss')


plt.plot(times_step, np.array(Heat_demand[0:H_day]), 'k--', label='Q demand')
plt.plot(times_step, stor_l[0:H_day], 'b--', label='Storage level')
plt.legend()

Fig_E=plt.figure()
plt.xticks(times_step, list(times_step))
plt.ylabel("Electric Energy [kWh]")
plt.title("Electricity balance")
plt.bar(times_step, el_purch[0:H_day], bottom=PV_gen[0:H_day]+res["El_prod_ICE1"][0:H_day]+ res["El_prod_ICE2"][0:H_day] +res["El_prod_ICE3"][0:H_day], label='el purch')
plt.bar(times_step, res["El_prod_ICE1"][0:H_day], bottom=PV_gen[0:H_day]+ res["El_prod_ICE2"][0:H_day]+res["El_prod_ICE3"][0:H_day], label='el prod ICE1')
plt.bar(times_step, res["El_prod_ICE2"][0:H_day], bottom=PV_gen[0:H_day]+res["El_prod_ICE3"][0:H_day], label='el prod ICE2')
plt.bar(times_step, res["El_prod_ICE3"][0:H_day], bottom=PV_gen[0:H_day], label='el prod ICE3')
plt.bar(times_step, PV_gen[0:H_day], label='PV output')
plt.bar(times_step, el_sold[0:H_day], label='el sold')

plt.plot(times_step, np.array(EE_demand[0:H_day]), 'r--', label='El demand')
plt.legend()
plt.show()