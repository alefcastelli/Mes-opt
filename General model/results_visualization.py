# Analizing and Plotting results

from optimization_model_concrete import *

model.solutions.load_from(results)
In=[]
z=[]
Out=[]
Out_us=[]
Out_diss=[]
Out_Res=[]
Net_exch=[]
Net_rev=[]
s=[]
SOC=[]
stor_net=[]
stor_charge=[]
stor_disch=[]
delta_on=[]
delta_off=[]
z_design=[]
z_design_stor=[]
x_design=[]
x_design_stor=[]
gamma=[]
gamma_stor=[]
c=[]
b=[]
b_stor=[]
Cinv=[]
Cinv_stor=[]
beta=[]
psi=[]
Res_area=[]

#var_list=[z_design, z_design_stor, z , delta_on, delta_off, b, b_stor, c, s, x_design, x_design_stor, Res_area, beta, psi, In, Out,
#        Out_diss, Out_us, Out_Res, Net_exch, SOC, stor_net, stor_charge, stor_disch, gamma, gamma_stor, Cinv, Cinv_stor, Net_rev]
pws_Boiler_0=[]
pws_Boiler_1=[]
pws_Boiler_2=[]
pws_ICE_0=[]
pws_ICE_1=[]
pws_ICE_2=[]
pws_HP_0=[]
pws_HP_1=[]
pws_HP_2=[]
pws_CC_0=[]
pws_CC_1=[]
pws_CC_2=[]
pws_TES=[]
pws_BESS=[]
pws_CES=[]
var_list=[z_design, z_design_stor, z , delta_on, delta_off, c, s, x_design, x_design_stor, Res_area, beta, psi, In, Out,
        Out_diss, Out_us, Out_Res, Net_exch, SOC, stor_net, stor_charge, stor_disch, Cinv, Cinv_stor, Net_rev,
          pws_Boiler_0, pws_Boiler_1, pws_Boiler_2, pws_ICE_0, pws_ICE_1, pws_ICE_2, pws_HP_0, pws_HP_1, pws_HP_2, pws_CC_0,
          pws_CC_1, pws_CC_2, pws_TES, pws_BESS, pws_CES]
i = 0
for v in model.component_objects(Var, active=True):
    print ("Variable",v)
    for index in v:
        print('  ', index, value(v[index]))
        var_list[i].append(value(v[index]))
    i=i+1


n_goods=len(model.Goods)
n_machines=len(model.Machines)
n_machines_ExtCons=len(model.Machines_ExtCons)
n_machines_IntCons=len(model.Machines_IntCons)
n_machines_diss=len(model.Machines_diss)


def Results_to_DF(res, n_machines, n_slots, Machines_set  ):
    frames=[]
    for i in range(n_machines):
        df=pd.DataFrame(res[i])
        frames.append(df)
    df_res=pd.concat(frames, ignore_index=True)
    df_res=df_res.transpose()

    name_list=[]
    for i in Machines_set.keys():
        for s in range(n_slots):
            name_list.append(i + ' s{}'.format(s+1))

    columns_name={}
    for i in range(len(df_res.columns)):
        columns_name[i]=name_list[i]
    df_res.rename(columns=columns_name, inplace=True)

    return df_res #.loc[:, (df_res != 0).any(axis=0)]  #<-- removes the zero-columns

# MAKE INPUT TO DATAFRAME
x_design=np.array(x_design).reshape(n_slots, n_machines)
In=np.array(In).reshape(n_machines, n_slots, T)
df_In=Results_to_DF(In, n_machines, n_slots, model.Machines)

# MAKE OUTPUT TO DATAFRAME
Out=np.array(Out).reshape(n_machines, n_slots, T*n_goods)
df_Out=Results_to_DF(Out, n_machines, n_slots, model.Machines)

if n_machines_diss > 0:
    Out_us=np.array(Out_us).reshape(n_machines_diss, n_slots, n_goods*T)
    df_Us=Results_to_DF(Out_us, n_machines_diss, n_slots, model.Machines_diss)
    Out_diss=-np.array(Out_diss).reshape(n_machines_diss, n_slots, n_goods*T)
    df_Diss=Results_to_DF(Out_diss, n_machines_diss, n_slots, model.Machines_diss)
Out_Res=np.array(Out_Res).reshape(len(model.Machines_Res), T*n_goods)

df_Res=pd.DataFrame(Out_Res.transpose(), columns=model.Machines_Res.keys())
Net_exch=-np.array(Net_exch).reshape(n_goods, T)
df_Net=pd.DataFrame(Net_exch.transpose(), columns=["El grid", "Heat net"])#, "Cold net"]) #columns=list(model.Goods))

df_Out_El=df_Out.iloc[0:T, :].reset_index(drop=True)
df_Out_Heat=df_Out.iloc[T:2*T, :].reset_index(drop=True)
df_Out_Cold=df_Out.iloc[2*T:3*T, :].reset_index(drop=True)

# initializing the final dataframe for each good
df_El=df_Out_El
df_Heat=df_Out_Heat
df_Cold=df_Out_Cold

# joining diss column to each dataframe
if n_machines_diss > 0:
    df_El=df_El.join(df_Diss.iloc[0:T, :].reset_index(drop=True), lsuffix='_gen', rsuffix='_diss')
    df_Heat=df_Heat.join(df_Diss.iloc[T:2*T, :].reset_index(drop=True), lsuffix='_gen', rsuffix='_diss')
    df_Cold=df_Cold.join(df_Diss.iloc[2*T:3*T, :].reset_index(drop=True), lsuffix='_gen', rsuffix='_diss')

df_El=df_El.join(df_Res.iloc[0:T, :].reset_index(drop=True))
df_Heat=df_Heat.join(df_Res.iloc[T:2*T, :].reset_index(drop=True))
df_Cold=df_Cold.join(df_Res.iloc[2*T:3*T, :].reset_index(drop=True))

df_El=df_El.join(df_Net['El grid'])
df_Heat=df_Heat.join(df_Net['Heat net'])
#df_Cold=df_Cold.join(df_Net['Cold net'])

SOC=np.array(SOC).reshape(len(model.Storages), T)
stor_net=np.array(stor_net).reshape(len(model.Storages), T)
stor_charge=-np.array(stor_charge).reshape(len(model.Storages), T)
stor_disch=np.array(stor_disch).reshape(len(model.Storages), T)
df_SOC=pd.DataFrame(SOC.transpose(), columns=model.Storages.keys())
df_storNet=pd.DataFrame(stor_net.transpose(), columns=model.Storages.keys())
df_charge=pd.DataFrame(stor_charge.transpose(), columns=model.Storages.keys())
df_disch=pd.DataFrame(stor_disch.transpose(), columns=model.Storages.keys())

for stor in model.Storages.keys():
    if 'El' in Storage_parameters[stor]['good']:
        df_El=df_El.join(df_SOC[stor], rsuffix='_SOC')
        df_El=df_El.join(df_storNet[stor], rsuffix='_net')
        #df_El = df_El.join(df_charge[el_stor], rsuffix='_charge')
        #df_El = df_El.join(df_disch[el_stor], rsuffix='_disch')
    if 'Heat' in Storage_parameters[stor]['good']:
        df_Heat=df_Heat.join(df_SOC[stor], rsuffix='_SOC')
        df_Heat=df_Heat.join(df_storNet[stor], rsuffix='_net')
        #df_Heat = df_Heat.join(df_charge[el_stor], rsuffix='_charge')
        #df_Heat = df_Heat.join(df_disch[el_stor], rsuffix='_disch')
    if 'Cold' in Storage_parameters[stor]['good']:
        df_Cold=df_Cold.join(df_SOC[stor], rsuffix='_SOC')
        df_Cold=df_Cold.join(df_storNet[stor], rsuffix='_net')
        #df_Cold = df_Cold.join(df_charge[el_stor], rsuffix='_charge')
        #df_Cold = df_Cold.join(df_disch[el_stor], rsuffix='_disch')

# filter across internal consumer machines
name_list = []
for i in model.Machines_IntCons.keys():
    for s in range(n_slots):
        name_list.append(i + ' s{}'.format(s + 1))
# add the internal consumptions to the df_El dataframe
for name in name_list:
    df_El = df_El.join(-df_In[name], rsuffix='_cons')


# Demand DataFrame
df_Demand=pd.DataFrame(EE_demand, columns=['El Demand'])
df_Demand["Heat Demand"]=Heat_demand
df_Demand["Cold Demand"]=Cold_demand

plt.figure()
ax1=df_Demand["El Demand"].plot(kind='line', color='g', linestyle='--', label='El demand', legend=True)
df_El["BESS"].plot(kind='line', ax=ax1, color='k', linestyle='--', label='Storage level', legend=True)
df_El= df_El.loc[:, (df_El != 0).any(axis=0)]  #<-- removes the zero-columns
if value(model.x_design_stor["BESS"]) == 0:
    df_El.plot(kind='bar', stacked=True, ax=ax1, rot=0, legend=True)
if value(model.x_design_stor["BESS"]) > 0:
    df_El.plot(kind='bar', y=df_El.columns.drop(["BESS"]), stacked=True, ax=ax1, rot=0, legend=True)
plt.xlabel("Timestep [h]")
plt.ylabel("Energy[kWh]")
plt.title("Electricity Production")

plt.figure()
ax2=df_Demand["Heat Demand"].plot(kind='line', color='r', linestyle='--', label='Heat demand', legend=True)
df_Heat["TES"].plot(kind='line', ax=ax2, color='k', linestyle='--', label='Storage level', legend=True)
df_Heat= df_Heat.loc[:, (df_Heat != 0).any(axis=0)]  #<-- removes the zero-columns
if value(model.x_design_stor["TES"]) == 0:
    df_Heat.plot(kind='bar', stacked=True, ax=ax2, rot=0)
if value(model.x_design_stor["TES"]) > 0:
    df_Heat.plot(kind='bar', y=df_Heat.columns.drop(["TES"]), stacked=True, ax=ax2, rot=0)
plt.xlabel("Timestep [h]")
plt.ylabel("Energy[kWh]")
plt.title("Heat Production")

'''
plt.figure()
df_Cold= df_Cold.loc[:, (df_Cold != 0).any(axis=0)]  #<-- removes the zero-columns
ax3=df_Demand["Cold Demand"].plot(kind='line', color='b', linestyle='--', label='Cold demand', legend=True)
df_Cold.plot(kind='bar', stacked=True, ax=ax3, legend=True, rot=0)
plt.xlabel("Timestep [h]")
plt.ylabel("Energy[kWh]")
plt.title("Cold Production")
'''