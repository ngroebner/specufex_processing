from sklearn.preprocessing import MinMaxScaler,Normalizer,RobustScaler
import numpy as np
import matplotlib.pyplot as plt
import os
def set_plot_prop():
    plt.ioff()
    mm2inch = lambda x: x/10./2.54
    # plt.rcParams['xtick.direction']= 'out'
    # plt.rcParams['ytick.direction']= 'out'
    plt.rcParams['axes.grid'] = True
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.rcParams['axes.labelsize'] = 24
    plt.rcParams['grid.color'] = 'k'
    plt.rcParams['grid.linestyle'] = ':'
    plt.rcParams['grid.linewidth'] = 0.75
    # plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 24
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['axes.linewidth'] = 2.
    plt.rcParams['figure.figsize'] = mm2inch(90*5),mm2inch(2./3*90*5)
    plt.rcParams["legend.handlelength"] = 1.
    plt.rcParams["legend.handletextpad"] = 0.15
    plt.rcParams["legend.borderpad"] = 0.15
    plt.rcParams["legend.labelspacing"] = 0.15
    cmap=plt.cm.get_cmap('RdYlBu')

def print_basics(f):
    print(f'Date {f["Date"][()]}')
    print(f'RockType {f["RockType"][()]}')
    print(f'Sample Diameter(mm) {f["SampleDiameter_mm"][()]}')
    print(f'Sample Length (mm) {f["SampleLength_mm"][()]}')
    print(f'Comfining Pressure (MPa) {f["P_conf_MPa"][()]}')
    print(f'Pore Pressure (MPa) {f["P_pore_MPa"][()]}')
    #print(f'Date {f["Date"][()]}')

def multi_plot(name_exp,time,y1,y2,linewidth=.1,label1='Dataset 1',label2='Dataset 2',alpha2=1,date=False,ylim=[None,None],filename=None,sci=True):
    #create figure
    fig, ax =plt.subplots(1,figsize=(15,10))
    # Plot y1 vs x in blue on the left vertical axis.
    plt.xlabel("Time (hr)")
    plt.ylabel(label1, color="b")
    plt.tick_params(axis="y", labelcolor="b")
    if sci :
        plt.ticklabel_format(axis="y", style="sci", scilimits=(-5,-4))
    plt.plot(time, y1, "b-", linewidth=linewidth)
    plt.title(name_exp+" Experiment")
    if date :
        ax = plt.gca()
        ax.xaxis_date()
    plt.ylim(ylim)
    plt.grid('False',color='b')
    #fig.autofmt_xdate(rotation=50)

    # Plot y2 vs x in red on the right vertical axis.
    plt.twinx()
    plt.ylabel(label2, color="r")
    plt.tick_params(axis="y", labelcolor="r")
    plt.plot(time, y2, "r-", linewidth=linewidth,alpha=alpha2)
    plt.grid('False',color='r')
    #plt.show()
    #To save your graph
    if filename :
        plt.savefig(filename)
    return 'Done'



def make_plots_exp_Dataset(new_data_def_Sampl,name_exp,time_window,max_val=140,Temperature=True):
    try :
        os.mkdir('Output_Plots')
    except :
        print('Output_Plots Exists')

    try :
        os.mkdir('Output_Plots/'+name_exp)
    except :
        print(f'Output_Plots/{name_exp} Exists')
    multi_plot(name_exp,new_data_def_Sampl.time_hr,new_data_def_Sampl['Sig_diff_MPa'],new_data_def_Sampl['AE_rate_count'],linewidth=1,label1='Differential Stress (MPa)'
               ,label2=r'AE Rate (min$^{-1}$)',alpha2=0.6,filename='Output_Plots/'+name_exp+'/'+name_exp+'_AE_Stress_'+str(time_window)+'s.pdf')
    plt.xlim([-2,max_val])

    multi_plot(name_exp,new_data_def_Sampl.time_hr,new_data_def_Sampl['stress_rate'],new_data_def_Sampl['AE_rate_count'],linewidth=1,label1='Differential Stress Rate (MPa/min)'
               ,label2=r'AE Rate (min$^{-1}$)',alpha2=0.6,filename='Output_Plots/'+name_exp+'/'+name_exp+'_AE_StressRate_'+str(time_window)+'s.pdf')
    plt.xlim([-2,max_val])
    ##############################################################################

    multi_plot(name_exp,new_data_def_Sampl.time_hr,new_data_def_Sampl['strain_rate'],new_data_def_Sampl['AE_rate_count'],linewidth=1,label1='Axial Strain Rate (1/min)'
               ,label2=r'AE Rate (min$^{-1}$)',alpha2=0.6,filename='Output_Plots/'+name_exp+'/'+name_exp+'_AE_Strain_'+str(time_window)+'s.pdf')
    plt.xlim([-2,max_val])

    plt.figure()
    plt.scatter(new_data_def_Sampl['strain_rate'],new_data_def_Sampl['AE_rate_count'],s=30,c=new_data_def_Sampl['Sig_diff_MPa'],cmap=plt.cm.RdYlBu)
    plt.colorbar(label='Differential Stress (MPa)')
    plt.xlim([-0.00002,0.0004])
    plt.ticklabel_format(axis="x", style="sci", scilimits=(-5,-4))
    plt.ylabel('AE rate (1/min)')
    plt.xlabel('Strain Rate (1/min)')
    plt.savefig('Output_Plots/'+name_exp+'/'+name_exp+'_AE_Strain_CrossPlot_'+str(time_window)+'s_.pdf')

    if Temperature == True :
        plt.figure()
        plt.plot(new_data_def_Sampl.Sig_diff_MPa,new_data_def_Sampl.Temp,'k')
        plt.xlabel('Differential Stress (MPa)')
        plt.ylabel('Temperature(C)')
        plt.savefig('Output_Plots/'+name_exp+'/'+name_exp+'_Stress_Temp_'+str(time_window)+'s_.pdf')

    plt.figure()
    plt.plot(new_data_def_Sampl.Sig_diff_MPa,new_data_def_Sampl.strain_rate,'ko',alpha=0.5)
    plt.xlabel('Differential Stress (MPa)')
    plt.ylabel('Strain Rate (1/min)')
    plt.savefig('Output_Plots/'+name_exp+'/'+name_exp+'_Stress_StrainRate_'+str(time_window)+'s_.pdf')

    if Temperature == True :
        multi_plot(name_exp,new_data_def_Sampl.time_hr,new_data_def_Sampl.Sig_diff_MPa,new_data_def_Sampl.Temp,linewidth=2,label1='Differential Stress (MPa)',label2='Temperature (C)',alpha2=1,sci=False,
                  filename='Output_Plots/'+name_exp+'/'+name_exp+'_DiffStress_Temp_'+str(time_window)+'s.pdf')


    plt.figure()
    plt.plot(new_data_def_Sampl.Strain_ax,new_data_def_Sampl.Sig_diff_MPa,'k')
    plt.ylabel('Differential Stress (MPa)')
    plt.xlabel('Strain')
    plt.savefig('Output_Plots/'+name_exp+'/'+name_exp+'_Stress_Strain_'+str(time_window)+'s_.pdf')

    plt.figure()
    plt.plot(new_data_def_Sampl.strain_rate,new_data_def_Sampl.Sig_diff_MPa,'ko',alpha=0.4)
    plt.ylabel('Differential Stress (MPa)')
    plt.xlabel('Strain Rate (1/min)')
    plt.savefig('Output_Plots/'+name_exp+'/'+name_exp+'_Stress_StrainRate_'+str(time_window)+'s_.pdf')

    multi_plot(name_exp,new_data_def_Sampl.time_hr,new_data_def_Sampl.Sig_diff_MPa,new_data_def_Sampl['Strain_ax'],linewidth=2,label1='Differential Stress (MPa)',label2='Strain',alpha2=1,sci=False,
               filename='Output_Plots/'+name_exp+'/'+name_exp+'_DiffStress_Strain_'+str(time_window)+'s.pdf')
    plt.xlim([-2,max_val])

    multi_plot(name_exp,new_data_def_Sampl.time_hr,new_data_def_Sampl.Sig_diff_MPa,new_data_def_Sampl['strain_rate'],linewidth=2,label1='Differential Stress (MPa)',
               label2='Strain Rate (1/min)',alpha2=1,sci=False,
               filename='Output_Plots/'+name_exp+'/'+name_exp+'_DiffStress_StrainRate_'+str(time_window)+'s.pdf')
    plt.xlim([-2,max_val])
    return 'Done'


def generate_data(new_data,new_data_def,time_window_val,start_date,end_date):
    time_window = str(time_window_val) +'s'
    new_data_1min = new_data.resample(time_window,closed='left', label='right').sum()
    new_data_1min = new_data_1min.loc[(new_data_1min.index >=start_date) & (new_data_1min.index <=end_date),:]

    new_data_1min_def = new_data_def.resample(time_window,closed='left', label='right').mean()
    new_data_1min_def = new_data_1min_def.loc[(new_data_1min_def.index >=start_date) & (new_data_1min_def.index <=end_date),:]

    new_data_1min_def['AE_rate_count'] = new_data_1min['count']
    new_data_1min_def['strain_rate'] =  new_data_1min_def['Strain_ax'].diff().fillna(0)/(time_window_val/60.) #1/min
    new_data_1min_def['stress_rate'] = new_data_1min_def['Sig_diff_MPa'].diff().fillna(0)/(time_window_val/60.) #1/min
    new_data_1min_def['AE_rate_count_rate'] = new_data_1min_def['AE_rate_count'].diff().fillna(0)/(time_window_val/60.) #1/min
    new_data_1min_def = work_rate(new_data_1min_def)
    return new_data_1min_def


def work_rate(df) :
    scaled_sig = MinMaxScaler().fit_transform(df['Sig_diff_MPa'].values.reshape(-1,1)).flatten()
    scaled_strain = MinMaxScaler().fit_transform(df['Strain_ax'].values.reshape(-1,1)).flatten()
    scaled_strain_rate = MinMaxScaler().fit_transform(df['strain_rate'].values.reshape(-1,1)).flatten()
    scaled_AE_rate_count = MinMaxScaler().fit_transform(df['AE_rate_count'].values.reshape(-1,1)).flatten()
    df['Sig_AE_rate'] = scaled_sig*scaled_AE_rate_count
    df['strain_rate_AE_rate'] = scaled_strain_rate*scaled_AE_rate_count
    df['strain_AE_rate'] = scaled_strain*scaled_AE_rate_count
    df['Sig_strain'] = scaled_sig*scaled_strain
    df['Sig_strain_rate'] = scaled_sig*scaled_strain_rate
    df['Strain_strain_rate'] = scaled_strain*scaled_strain_rate
    #df['Sig_diff_MPa_PlusOne'] = df['Sig_diff_MPa'].shift(periods=-1).fillna(df['Sig_diff_MPa'].max())
    return df


def make_file(new_data_def_Sampl,scaled_data2,path,noise_amp=0.05):
    data_all_scaled_Robust = new_data_def_Sampl.copy(deep=True)
    data_all_scaled_Robust[:] = scaled_data2
    data_all_scaled_Robust.to_csv(path+'.csv',index=False)

    scaled_data2 = scaled_data2 + scaled_data2.mean(axis=0)*(np.random.rand(scaled_data2.shape[0],scaled_data2.shape[1]))*noise_amp
    data_all_scaled_Robust[:] = scaled_data2
    data_all_scaled_Robust.to_csv(path+'_'+str(noise_amp)+'.csv',index=False)

#     '/home/tmittal/Causality/Basalt_Data/dataforcausalitytests_'+name_exp+'/'+time_agg+'/Basalt_data4tcdf_NonStationary_'+time_agg+'_'+name_scalar+'.csv'

