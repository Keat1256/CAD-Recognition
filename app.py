import os
import matplotlib.pyplot as plt
import json
import re
import pandas as pd
from st_aggrid import AgGrid, GridUpdateMode
from st_aggrid.grid_options_builder import GridOptionsBuilder
import math 
from datetime import datetime
import Levenshtein
from itertools import product
from zipfile import ZipFile
from io import BytesIO
from base64 import b64encode
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import cv2
import tensorflow as tf
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if any(gpu_devices):
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from doctr.utils.visualization import visualize_page
import streamlit as st
import streamlit_authenticator as stauth
from streamlit_option_menu import option_menu
import streamlit_modal as modal
import streamlit.components.v1 as components
import sqlite3 

conn = sqlite3.connect("data.db", check_same_thread=False)
cur = conn.cursor()

st.set_page_config(
    page_title="Dashboard",
    layout="wide",
)

hide_st_style="""
            <style>
            #MainMenu {visibility:hidden;}
            footer {visibility:hidden;}
            div.block-container {padding-top:3rem;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

def main():
    
    cols = st.columns((1, 1, 1, 1))
    
    st.sidebar.title("Document selection")
    
    st.set_option('deprecation.showfileUploaderEncoding', False)
    
    uploaded_file = st.sidebar.file_uploader("Upload files", type=['pdf'])

    if uploaded_file is not None:
        DVP = st.sidebar.text_input("DVP:", uploaded_file.name[:-4])
        hide = st.sidebar.checkbox('Hide recognition data picture',value=True)

        if uploaded_file.name.endswith('.pdf'):
            doc = DocumentFile.from_pdf(uploaded_file.read())

        if not hide:
            cols[0].subheader("Input page")
            cols[1].subheader("Segmentation heatmap")
            cols[2].subheader("OCR output")
            cols[3].subheader("Page reconstitution")
            cols[0].image(doc[0])
    
    det_arch = "db_resnet50"
    reco_arch = "crnn_vgg16_bn"



    st.sidebar.write('\n')

    if st.sidebar.button("Drawing Recognition"):

        if uploaded_file is None:
            st.sidebar.write("Please upload a document")

        else:
            with st.spinner('Loading model...'):
                predictor = ocr_predictor(
                    det_arch, reco_arch, pretrained=True,
                    assume_straight_pages=(det_arch != "linknet_resnet18_rotation")
                )

            with st.spinner('Analyzing...'):

                # Forward the image to the model
                processed_batches = predictor.det_predictor.pre_processor([doc[0]])
                out = predictor.det_predictor.model(processed_batches[0], return_model_output=True)
                seg_map = out["out_map"]
                seg_map = tf.squeeze(seg_map[0, ...], axis=[2])
                seg_map = cv2.resize(seg_map.numpy(), (doc[0].shape[1], doc[0].shape[0]),
                                     interpolation=cv2.INTER_LINEAR)
                # Plot the raw heatmap
                if not hide:
                    fig, ax = plt.subplots()
                    ax.imshow(seg_map)
                    ax.axis('off')
                    cols[1].pyplot(fig)

                # Plot OCR output
                out = predictor([doc[0]])
                if not hide:
                    fig = visualize_page(out.pages[0].export(), doc[0], interactive=False)
                    cols[2].pyplot(fig)

                # Page reconsitution under input page
                page_export = out.pages[0].export()
                if not hide:
                    img = out.pages[0].synthesize()
                    cols[3].image(img, clamp=True)

                # Display result
                st.markdown("\nHere are your recognition results:")
                diameterdisct={"value":[],"geometry":[]}
                hardwaredisct={"value":[],"geometry":[]}
                quantitydisct={"value":[],"geometry":[]}
                for a in page_export['blocks'] :
                    for b in a['lines'] :
                        for c in b['words'] :
                            if re.match('^(?=.*[0-9])(?=.*[A-Z])(?=.*[-])([A-Z0-9_\"\'-]+)$',c['value']):
                                hardwaredisct['value'].append(re.sub('["\']', '', c['value']))
                                hardwaredisct['geometry'].append((c['geometry'][0][0],c['geometry'][0][1]))

                            if  re.match('^([09]{1,1}[1-9]{1,1}[A-Zrxk0-9-.:,/()\"\']{2,})$',c['value']):
                                diameterdisct['value'].append(c['value'])
                                diameterdisct['geometry'].append((c['geometry'][0][0],c['geometry'][0][1]))

                            if  re.match('^([(]{1,1}[0-9]{1,1}[A-Zrxk0-9-.:,/()\"\']{2,})$',c['value']) or re.match('^([0-9]{1,1}[X]{1,1}[A-Zrxk0-9-.:,/()\"\']{1,})$',c['value']):
                                quantitydisct['value'].append(re.sub('[A-Za-z"\'()]', '', c['value']))
                                quantitydisct['geometry'].append((c['geometry'][0][0],c['geometry'][0][1]))

                resultdict={"hardware":[],"diameter":[],"quantity":[]}
                for d in range(len(hardwaredisct['value'])):
                    conn = sqlite3.connect("data.db", check_same_thread=False)
                    cur = conn.cursor()
                    cur.execute("SELECT * FROM BOM WHERE (DVP = '%s' AND hardware_id = '%s') " %(DVP,hardwaredisct['value'][d]))
                    result=cur.fetchall()
                    conn.commit()
                    conn.close()
                    if len(result)>=1: 
                        smallest=1
                        smallestquantity=1
                        diameter=""
                        quantity=""
                        for e in range(len(diameterdisct['value'])):
                            georange=math.dist([hardwaredisct['geometry'][d][0],hardwaredisct['geometry'][d][1]],[diameterdisct['geometry'][e][0],diameterdisct['geometry'][e][1]])
                            if georange<smallest:
                                smallest=georange
                                if not re.match('(?=.{1,10}$)',diameterdisct['value'][e]):
                                    if re.match('(?=.*[xXY])',diameterdisct['value'][e]):
                                        p = re.compile("[xXY]")
                                        for m in p.finditer(diameterdisct['value'][e]):
                                            quantity=diameterdisct['value'][e][m.start()-1]
                                    else:
                                        quantity="1"
                                else:
                                    quantity="1"

                                diameter=re.sub('[A-Za-z"\'():.]', '', diameterdisct['value'][e])
                                diameter=diameter[0:4]
                                diameter= diameter[:2] + '.' + diameter[2:]
                                if(diameter[0]=='9'):
                                        diameter=diameter[1:5]
                                diameter=float(diameter)
                        resultdict['hardware'].append(hardwaredisct['value'][d])
                        resultdict['diameter'].append(diameter)
                        resultdict['quantity'].append(quantity)


                    else:
                        conn = sqlite3.connect("data.db", check_same_thread=False)
                        cur = conn.cursor()
                        cur.execute("SELECT * FROM BOM WHERE (DVP = '%s') " %(DVP))
                        resultb=cur.fetchall()
                        conn.commit()
                        conn.close()
                        highest=0.5
                        almostsame=""
                        for databaselen in range(len(resultb)):
                            similarity=Levenshtein.ratio(hardwaredisct['value'][d], resultb[databaselen][2])
                            if similarity>highest:
                                highest=similarity
                                almostsame=resultb[databaselen][2]
                        if almostsame!="":
                            comhardware=hardwaredisct['value'][d].replace("o","O")
                            comhardware=hardwaredisct['value'][d].replace("0","O")
                            combination=list(filler(comhardware, "O", "0"))
                            for clen in range(len(combination)):
                                similarity=Levenshtein.ratio(combination[clen], almostsame)
                                if similarity==1:
                                    hardwaredisct['value'][d]=combination[clen]
                            smallest=1
                            smallestquantity=1
                            diameter=""
                            quantity=""
                            for e in range(len(diameterdisct['value'])):
                                georange=math.dist([hardwaredisct['geometry'][d][0],hardwaredisct['geometry'][d][1]],[diameterdisct['geometry'][e][0],diameterdisct['geometry'][e][1]])
                                if georange<smallest:
                                    smallest=georange
                                    if not re.match('(?=.{1,10}$)',diameterdisct['value'][e]):
                                        if re.match('(?=.*[xXY])',diameterdisct['value'][e]):
                                            p = re.compile("[xXY]")
                                            for m in p.finditer(diameterdisct['value'][e]):
                                                quantity=diameterdisct['value'][e][m.start()-1]
                                        else:
                                            quantity="1"
                                    else:
                                        quantity="1"

                                    diameter=re.sub('[A-Za-z"\'():.]', '', diameterdisct['value'][e])
                                    diameter=diameter[0:4]
                                    diameter= diameter[:2] + '.' + diameter[2:]
                                    if(diameter[0]=='9'):
                                        diameter=diameter[1:5]
                                    diameter=float(diameter)

                            resultdict['hardware'].append(hardwaredisct['value'][d])
                            resultdict['diameter'].append(diameter)
                            resultdict['quantity'].append(quantity)


                            
                for g in range(len(quantitydisct['value'])):
                    smallestquantity=1
                    hardware=""
                    for h in range(len(resultdict['hardware'])):
                        georange=math.dist([hardwaredisct['geometry'][h][0],hardwaredisct['geometry'][h][1]],[quantitydisct['geometry'][g][0],quantitydisct['geometry'][g][1]])
                        if georange<smallestquantity:
                            smallestquantity=georange
                            hardware=hardwaredisct['value'][h]
                    if resultdict['hardware'].count(hardware)>0:
                        index=resultdict['hardware'].index(hardware)
                        resultdict['quantity'][index]=quantitydisct['value'][g]

                
                finalresult={"hardware":[],"diameter":[],"quantity":[]}
                conn = sqlite3.connect("data.db", check_same_thread=False)
                cur = conn.cursor()
                cur.execute("SELECT * FROM BOM WHERE (DVP = '%s') " %(DVP))
                result=cur.fetchall()
                conn.commit()
                conn.close()
                counter=0
                if len(result)>=1: 
                    conn = sqlite3.connect("data.db", check_same_thread=False)
                    cur = conn.cursor()
                    cur.execute("""CREATE TABLE IF NOT EXISTS recognition_history(history_id INTEGER PRIMARY KEY AUTOINCREMENT, user TEXT, time TEXT, file TEXT);""")
                    cur.execute("SELECT * FROM recognition_history WHERE (file = '%s') " %(DVP))
                    fresult=cur.fetchall()
                    if len(fresult)>=1:
                        
                        while(len(fresult)>=1):
                            counter+=1
                            cur.execute("SELECT * FROM recognition_history WHERE (file = '%s') " %(DVP+"("+str(counter)+")"))
                            fresult=cur.fetchall()
                        cur.execute("INSERT INTO recognition_history(user,time,file) VALUES (?,?,?)", (st.session_state["username"],datetime.now(),(DVP+"("+str(counter)+")")))
                    else:
                        cur.execute("INSERT INTO recognition_history(user,time,file) VALUES (?,?,?)", (st.session_state["username"],datetime.now(),DVP))
                    conn.commit()
                    conn.close()

                    result_table = '''
                        <table style="text-align: center;">
                            <tr>
                                <th colspan="2">Hardware</th>
                                <th colspan="2">Diameter</th>
                                <th colspan="2">Quantity</th>
                                <th colspan="2">Result</th>
                            </tr>
                            <tr>
                                <th>BOM Result</th>
                                <th>Recognition Result</th>
                                <th>Diameter Spec Result</th>
                                <th>Recognition Result</th>
                                <th>BOM Result</th>
                                <th>Recognition Result</th>
                                <th>Match</th>
                                <th>Remark</th>
                            </tr>
                        '''
                    for n in range(len(result)):
                        highest=0.5
                        almostsame=""
                        for m in range(len(resultdict["hardware"])):
                            similarity=Levenshtein.ratio(result[n][2], resultdict["hardware"][m])
                            if similarity>highest:
                                highest=similarity
                                hardware=resultdict["hardware"][m]
                                diameter=resultdict["diameter"][m]
                                quantity=resultdict["quantity"][m]

                        conn = sqlite3.connect("data.db", check_same_thread=False)
                        cur = conn.cursor()
                        cur.execute("SELECT * FROM hardware_diameter WHERE (hardware_id = '%s') " %(result[n][2]))
                        result2=cur.fetchall()
                        conn.commit()
                        conn.close()

                        if (result[n][2]==hardware) and (float("{:.2f}".format(result2[0][1]))-diameter<=result2[0][3]) and (float("{:.2f}".format(diameter-result2[0][1]))<=result2[0][2]) and (str(result[n][3])==quantity):
                            result_table+='''<tr style="background-color:#bfe3b4">
                                    <td>'''+result[n][2]+'''</td>
                                    <td>'''+hardware+'''</td>
                                    <td>'''+str(result2[0][1])+'''</td>
                                    <td>'''+str(diameter)+'''</td>
                                    <td>'''+str(result[n][3])+'''</td>
                                    <td>'''+quantity+'''</td>
                                    <td>Yes</td>
                                    <td>-</td>
                                </tr>'''
                            conn = sqlite3.connect("data.db", check_same_thread=False)
                            cur = conn.cursor()
                            cur.execute("""CREATE TABLE IF NOT EXISTS recognition_history_result(result_id INTEGER PRIMARY KEY AUTOINCREMENT, file TEXT, hardware_bom_result TEXT, hardware_recognition_result TEXT, diameter_diameter_spec_result TEXT, diameter_recognition_result TEXT, quantity_bom_result TEXT, quantity_recognition_result TEXT, match TEXT, remark TEXT);""")
                            if counter>0:
                                cur.execute("INSERT INTO recognition_history_result(file,hardware_bom_result,hardware_recognition_result,diameter_diameter_spec_result,diameter_recognition_result,quantity_bom_result,quantity_recognition_result,match,remark) VALUES (?,?,?,?,?,?,?,?,?)", ((DVP+"("+str(counter)+")"),result[n][2],hardware,str(result2[0][1]),diameter,str(result[n][3]),quantity,"Yes","-"))
                            else:
                                cur.execute("INSERT INTO recognition_history_result(file,hardware_bom_result,hardware_recognition_result,diameter_diameter_spec_result,diameter_recognition_result,quantity_bom_result,quantity_recognition_result,match,remark) VALUES (?,?,?,?,?,?,?,?,?)", ((DVP),result[n][2],hardware,str(result2[0][1]),diameter,str(result[n][3]),quantity,"Yes","-"))

                            conn.commit()
                            conn.close()
                        else:
                            remark = ""
                            result_table += '''<tr style="background-color:#f2b195 ">
                                        <td>''' + result[n][2] + '''</td>
                                        <td>''' + hardware + '''</td>
                                        <td>''' + str(result2[0][1]) + '''</td>
                                        <td>''' + str(diameter) + '''</td>
                                        <td>''' + str(result[n][3]) + '''</td>
                                        <td>''' + quantity + '''</td>
                                        <td>No</td>
                                        <td>'''
                            if (result[n][2]!=hardware):         
                                result_table+='''Hardware code, '''
                                remark+='''Hardware code, '''
                            if (float("{:.2f}".format(result2[0][1]))-diameter>result2[0][3]) or (float("{:.2f}".format(diameter-result2[0][1]))>result2[0][2]):         
                                result_table+='''Diameter, '''
                                remark+='''Diameter, '''
                            if (str(result[n][3])!=quantity):         
                                result_table+='''Quantity '''
                                remark+='''Quantity, '''
                            remark+='''not match'''
                            conn = sqlite3.connect("data.db", check_same_thread=False)
                            cur = conn.cursor()
                            cur.execute("""CREATE TABLE IF NOT EXISTS recognition_history_result(result_id INTEGER PRIMARY KEY AUTOINCREMENT, file TEXT, hardware_bom_result TEXT, hardware_recognition_result TEXT, diameter_diameter_spec_result TEXT, diameter_recognition_result TEXT, quantity_bom_result TEXT, quantity_recognition_result TEXT, match TEXT, remark TEXT);""")
                            if counter>0:
                                cur.execute("INSERT INTO recognition_history_result(file,hardware_bom_result,hardware_recognition_result,diameter_diameter_spec_result,diameter_recognition_result,quantity_bom_result,quantity_recognition_result,match,remark) VALUES (?,?,?,?,?,?,?,?,?)", ((DVP+"("+str(counter)+")"),result[n][2],hardware,str(result2[0][1]),diameter,str(result[n][3]),quantity,"No",remark))
                            else:
                                cur.execute("INSERT INTO recognition_history_result(file,hardware_bom_result,hardware_recognition_result,diameter_diameter_spec_result,diameter_recognition_result,quantity_bom_result,quantity_recognition_result,match,remark) VALUES (?,?,?,?,?,?,?,?,?)", ((DVP),result[n][2],hardware,str(result2[0][1]),diameter,str(result[n][3]),quantity,"No",remark))

                            conn.commit()
                            conn.close()
                            result_table+='''not match</td>
                                </tr>'''
                
                    result_table+='''</table>'''
                
                    st.markdown(result_table, unsafe_allow_html=True)
                    

                else:
                    st.warning("No Data of "+DVP)

def filler(word, from_char, to_char):
    options = [(c,) if c != from_char else (from_char, to_char) for c in word]
    return(''.join(o) for o in product(*options))

def BOM():
    st.sidebar.title("Upload BOM(DVP, hardware, quantity)")
    uploaded_file = st.sidebar.file_uploader("Choose your files", type=['xlsx', 'xls'])
    
    if st.sidebar.button("Upload data"):
        if uploaded_file is None:
            st.error("Please upload a document")
        else:
            reDVP=[]
            reHardware=[]
            reRow=[]
            quanHardware=[]
            quanDVP=[]
            quanRow=[]
            nhRow=[]
            ndRow=[]
            preDVP=""
            df=pd.read_excel(uploaded_file, header=None, skiprows=[0])
            col_num=checkfile(df)
            if col_num==3:
                for y in range(len(df[0])):
                    if pd.isna(df[0][y]):
                        df[0][y]=preDVP
                    else:
                        preDVP=df[0][y]
                    if isfloat(str(df[2][y])) and float(df[2][y])>0 and not pd.isna(df[1][y]) and not df[0][y]=="":
                        if str(df[2][y]).isnumeric():
                            conn = sqlite3.connect("data.db", check_same_thread=False)
                            cur = conn.cursor()
                            cur.execute("""CREATE TABLE IF NOT EXISTS BOM(BOM_id INTEGER PRIMARY KEY AUTOINCREMENT, DVP TEXT, hardware_id TEXT, quantity INTEGER);""")
                            cur.execute("SELECT * FROM BOM WHERE (DVP = '%s' AND hardware_id = '%s') " %(df[0][y],df[1][y]))
                            result=cur.fetchall()
                            if len(result)>=1:
                                cur.execute("UPDATE BOM SET quantity = ? WHERE DVP = ? AND hardware_id = ?", (int(df[2][y]),df[0][y],df[1][y]))
                                reDVP.append(df[0][y])
                                reHardware.append(df[1][y])
                                reRow.append(y+2)

                            else:
                                cur.execute("INSERT INTO BOM(DVP,hardware_id,quantity) VALUES (?,?,?)", (df[0][y],df[1][y],int(df[2][y])))
                            conn.commit()
                            conn.close()
                        else:
                            quanDVP.append(df[0][y])
                            quanHardware.append(df[1][y])
                            quanRow.append(y+2)
                            
                    if not isfloat(str(df[2][y])) or float(df[2][y])<=0 or pd.isna(df[2][y]):
                        quanDVP.append(df[0][y])
                        quanHardware.append(df[1][y])
                        quanRow.append(y+2)


                    if pd.isna(df[1][y]):
                        nhRow.append(y+2)
                
                    if df[0][y]=="":
                        ndRow.append(y+2)
                
                if len(reDVP)>0 and len(reRow)>0 and len(reHardware)>0:
                    repeat='''
                    Repeated variable:'''
                    for re in range(len(reDVP)):
                        repeat+='''
                        In Row: '''+str(reRow[re])+''' DVP: '''+str(reDVP[re])+''', Hardware: '''+str(reHardware[re])
                    st.warning('''Repeated hardware code in same DVP detected. Only latest quantity will be upload
                    '''+repeat)
                
                if len(quanDVP)>0 and len(quanRow)>0 and len(quanHardware)>0:
                    invalid='''
                    Ignored variable:'''
                    for re in range(len(quanDVP)):
                        invalid+='''
                        In Row: '''+str(quanRow[re])+''' DVP: '''+str(quanDVP[re])+''', Hardware: '''+str(quanHardware[re])
                    st.warning('''In Quantity column only can only accept number more than 0. Invalid variable in Quantity column will be ignored. 
                    '''+invalid)
                
                if len(nhRow)>0:
                    invalid='''
                    Row with empty hardware variable:'''
                    for re in range(len(nhRow)):
                        invalid+='''
                        In Row: '''+str(nhRow[re])
                    st.warning('''Empty value of hardware are not allowed to upload. 
                    '''+invalid)
                
                if len(ndRow)>0:
                    invalid='''
                    Row with empty DVP variable:'''
                    for re in range(len(ndRow)):
                        invalid+='''
                        In Row: '''+str(ndRow[re])
                    st.warning('''Empty value of DVP are not allowed to upload. 
                    '''+invalid)
                
                if len(reRow)==0 and len(reDVP)==0 and len(reHardware)==0 and len(quanRow)==0 and len(quanDVP)==0 and len(quanHardware)==0 and len(nhRow)==0 and len(ndRow)==0:
                    st.success("Upload successful!")
            
            else:
                st.warning("Wrong file uploaded!!!")


    st.sidebar.title("Add hardware detail")
    DVP = st.sidebar.text_input("DVP:")
    Hardware = st.sidebar.text_input("Hardware:")
    Quantity = st.sidebar.number_input("Quantity:",min_value=1,step=1,format="%i")
    if st.sidebar.button("Add"):
        if (str(DVP) and not str(DVP).isspace()) and (str(Hardware) and not str(Hardware).isspace()) and (str(Quantity) and not str(Quantity).isspace()):
            reDVP=[]
            reHardware=[]
            conn = sqlite3.connect("data.db", check_same_thread=False)
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS BOM(BOM_id INTEGER PRIMARY KEY AUTOINCREMENT, DVP TEXT, hardware_id TEXT, quantity INTEGER);""")
            cur.execute("SELECT * FROM BOM WHERE (DVP = '%s' AND hardware_id = '%s') " %(DVP,Hardware))
            result=cur.fetchall()
            if len(result)>=1:
                cur.execute("UPDATE BOM SET quantity = ? WHERE DVP = ? AND hardware_id = ?", (Quantity,DVP,Hardware))
                reDVP.append(DVP)
                reHardware.append(Hardware)
            else:
                cur.execute("INSERT INTO BOM(DVP,hardware_id,quantity) VALUES (?,?,?)", (DVP,Hardware,Quantity))

            conn.commit()
            conn.close()

            if len(reDVP)>0 and len(reHardware)>0:
                for re in range(len(reDVP)):
                    st.warning("Repeated hardware code: "+reHardware[re]+" in DVP: "+reDVP[re]+" detected. Only latest quantity will be upload")
            else:
                st.success("Upload successful!")
        else:
            st.error("Fail too add data!!! Empty value cannot upload.")

    conn = sqlite3.connect("data.db", check_same_thread=False)
    cur = conn.cursor()
    cur.execute("SELECT * FROM BOM")
    result=cur.fetchall()
    conn.commit()
    conn.close()
    if len(result)>=1:
        for de in range(len(result)):
            mylist=list(result[de])
            mylist.remove(result[de][0])
            result[de]=tuple(mylist)

        table=pd.DataFrame(result)
        BOMtable(table)
    else:
        st.write("NO DATA THERE!!!")

def BOMtable(a):
    placeholder = st.empty()
    a.columns=["DVP","Hardware","Quantity"]

    gd = GridOptionsBuilder.from_dataframe(a)
    gd.configure_pagination(enabled=True)
    gd.configure_default_column(groupable=True,editable=True)

    gd.configure_selection(selection_mode='multiple', use_checkbox=True)
    gridoptions = gd.build()

    col6, col7 =st.columns([1,0.3])
    
    grid_table = AgGrid(a,gridOptions=gridoptions,height=774,width='100%',udate_mode= GridUpdateMode.MODEL_CHANGED,fit_columns_on_grid_load=True,allow_unsafe_jscode=True)

    sel_row=grid_table["selected_rows"]
    nHardware=[]
    nDVP=[]
    nQuantity=[]
    nRow=[]
    with col2:
        if st.button("Delete"):
            for selectednum in range(len(sel_row)):
                conn = sqlite3.connect("data.db", check_same_thread=False)
                cur = conn.cursor()
                cur.execute("DELETE FROM BOM WHERE (DVP = '%s' AND hardware_id = '%s' AND quantity = '%s') " %(sel_row[selectednum]['DVP'],sel_row[selectednum]['Hardware'],sel_row[selectednum]['Quantity']))
                conn.commit()
                conn.close()
            st.experimental_rerun()

    with col3:
        if st.button("Update",key="1"):
            conn = sqlite3.connect("data.db", check_same_thread=False)
            cur = conn.cursor()
            cur.execute("SELECT * FROM BOM ")
            result=cur.fetchall()
            conn.commit()
            conn.close()
            
            for le in range(len(result)):
                if grid_table['data']['DVP'][le]!="" and not grid_table['data']['DVP'][le].isspace() and grid_table['data']['Hardware'][le]!="" and not grid_table['data']['Hardware'][le].isspace() and not pd.isna(grid_table['data']['Quantity'][le]) and isfloat(grid_table['data']['Quantity'][le]) and float(grid_table['data']['Quantity'][le])>0:
                    if (result[le][1]!=grid_table['data']['DVP'][le]) or (result[le][2]!=grid_table['data']['Hardware'][le]) or (result[le][3]!=grid_table['data']['Quantity'][le]):
                        
                        conn = sqlite3.connect("data.db", check_same_thread=False)
                        cur = conn.cursor()
                        cur.execute("SELECT * FROM BOM WHERE (DVP = '%s' AND hardware_id = '%s') " %(grid_table['data']['DVP'][le],grid_table['data']['Hardware'][le]))
                        re=cur.fetchall()
                        
                        if len(re)>=1:
                            for r in range(len(re)):
                                if result[le][0]==re[r][0]:
                                    cur.execute("UPDATE BOM SET DVP = ?, hardware_id = ?, quantity = ? WHERE BOM_id = ?", (grid_table['data']['DVP'][le],grid_table['data']['Hardware'][le],int(grid_table['data']['Quantity'][le]),result[le][0]))
                                else:
                                    cur.execute("UPDATE BOM SET quantity = ? WHERE DVP = ? AND hardware_id = ?", (int(grid_table['data']['Quantity'][le]),grid_table['data']['DVP'][le],grid_table['data']['Hardware'][le]))
                                    cur.execute("DELETE FROM BOM WHERE BOM_id = '%s' " %(result[le][0]))
                        else:
                            cur.execute("UPDATE BOM SET DVP = ?, hardware_id = ?, quantity = ? WHERE BOM_id = ?", (grid_table['data']['DVP'][le],grid_table['data']['Hardware'][le],int(grid_table['data']['Quantity'][le]),result[le][0]))

                        conn.commit()
                        conn.close()
                else: 
                    nDVP.append(result[le][1])
                    nHardware.append(result[le][2])
                    nQuantity.append(result[le][3])
                    nRow.append(le+1)
            
    with col6:
        if len(nDVP)>0 and len(nRow)>0 and len(nHardware)>0 and len(nQuantity)>0:
            invalid='''
            Will remain the same:'''
            for re in range(len(nDVP)):
                invalid+='''
                In Row: '''+str(nRow[re])+''' DVP: '''+str(nDVP[re])+''', Hardware: '''+str(nHardware[re]) +''', Quantity: '''+str(nQuantity[re])
            st.warning('''Empty or invalid value cannot be update 
            '''+invalid)

def history():
    
    conn = sqlite3.connect("data.db", check_same_thread=False)
    cur = conn.cursor()
    if st.session_state["name"]=="Admin":
        cur.execute("SELECT * FROM recognition_history ")
    else:
        cur.execute("SELECT * FROM recognition_history WHERE (user = '%s') " %(st.session_state["username"]))
    result=cur.fetchall()
    conn.commit()
    conn.close()
    if len(result)>=1:
        for de in range(len(result)):
            mylist=list(result[de])
            mylist.remove(result[de][0])
            result[de]=tuple(mylist)
        table=pd.DataFrame(result)
        historytable(table)
    else:
        st.write("NO DATA THERE!!!")

def historytable(a):
    placeholder = st.empty()
    a.columns=["User","Time","File"]

    gd = GridOptionsBuilder.from_dataframe(a)
    gd.configure_pagination(enabled=True)
    gd.configure_default_column(groupable=True)

    gd.configure_selection(selection_mode='multiple', use_checkbox=True)
    gridoptions = gd.build()

    grid_table = AgGrid(a,gridOptions=gridoptions,height=774,width='100%',udate_mode= GridUpdateMode.MODEL_CHANGED,fit_columns_on_grid_load=True,allow_unsafe_jscode=True)

    sel_row=grid_table["selected_rows"]
    with col3:
        if len(sel_row)>0:
            zipObj = ZipFile("Excel_Export.zip", "w")
            for selectednum in range(len(sel_row)):
                conn = sqlite3.connect("data.db", check_same_thread=False)
                cur = conn.cursor()
                cur.execute("SELECT * FROM recognition_history_result WHERE (file = '%s') " %(sel_row[selectednum]['File']))
                hresult=cur.fetchall()
                    
                conn.commit()
                conn.close()
                if len(hresult)>0:
                    for de in range(len(hresult)):
                        mylist=list(hresult[de])
                        mylist.remove(hresult[de][0])
                        mylist.remove(hresult[de][1])
                        hresult[de]=tuple(mylist)
                    export_excel=pd.DataFrame(hresult)
                    export_excel.columns=["BOM Result(hardware)","Recognition Result(hardware)","BOM Result(diameter)","Recognition Result(diameter)","BOM Result(quantity)","Recognition Result(quantity)","Match","Remark"]
                    file_name = str(sel_row[selectednum]['File'])+'.xlsx'
                    export_excel.to_excel(file_name,index=False)
                    zipObj.write(file_name)
            zipObj.close()
            ZipfileDotZip = "Excel_Export.zip"

            with open(ZipfileDotZip, "rb") as f:
                bytes = f.read()
                b64 = b64encode(bytes).decode()
                href='''<style>
                            a:link, a:visited {
                                border-radius: 5px;

                                background-color: white;
                                color: black;
                                border: 1px solid #d7d7da;
                                padding: 10px 20px;
                                text-align: center;
                                text-decoration: none;
                                display: inline-block;
                            }
                            a{
                                margin-bottom: 25px;

                            }
                            a:hover {
                                background-color: white;
                                border-color:#ff5555;
                                color:#ff5555;
                            }
                            a:focus{
                                border-color:#ff5555;
                                color:#ff5555;
                                box-shadow: 0 0 0 4px #ff9292;
                            }
                            a:active{
                                background-color: #ff4b4b;
                                border-color:#ff5555;
                                color:white;
                                box-shadow: 0 0 0 4px #ff9292;
                            }
                        </style>'''
                href += f"<a href=\"data:file/zip;base64,{b64}\" download='{ZipfileDotZip}.zip'>\
                    Export Excel\
                </a>"
            st.markdown(href, unsafe_allow_html=True)

        
        


def diameter():
    st.sidebar.title("Upload Diameter Spec(hardware,diameter)")
    uploaded_file = st.sidebar.file_uploader("Choose your files", type=['xlsx', 'xls'])
    
    if st.sidebar.button("Upload data"):
        if uploaded_file is None:
            st.error("Please upload a document")
        else:
            reRow=[]
            reHardware=[]
            diaHardware=[]
            diaRow=[]
            nhRow=[]
            df=pd.read_excel(uploaded_file, header=None, skiprows=[0])
            col_num=checkfile(df)
            if col_num==4:
                for y in range(len(df[0])):
                    if isfloat(str(df[1][y])) and float(df[1][y])>0 and not pd.isna(df[0][y]) and isfloat(str(df[2][y])) and float(df[2][y])>=0 and isfloat(str(df[3][y])) and float(df[3][y])>=0:
                        conn = sqlite3.connect("data.db", check_same_thread=False)
                        cur = conn.cursor()
                        cur.execute("""CREATE TABLE IF NOT EXISTS hardware_diameter(hardware_id TEXT PRIMARY KEY, diameter NUMERIC, upper_tolerance NUMERIC, lower_tolerance NUMERIC);""")
                        cur.execute("SELECT * FROM hardware_diameter WHERE (hardware_id = '%s') " %(df[0][y]))
                        result=cur.fetchall()
                        if len(result)>=1:
                            cur.execute("UPDATE hardware_diameter SET diameter = ?, upper_tolerance = ?, lower_tolerance = ? WHERE hardware_id = ?", (df[1][y],df[2][y],df[3][y],df[0][y]))
                            reRow.append(y+2)
                            reHardware.append(df[0][y])
                        else:
                            cur.execute("INSERT INTO hardware_diameter(hardware_id,diameter,upper_tolerance,lower_tolerance) VALUES (?,?,?,?)", (df[0][y],df[1][y],df[2][y],df[3][y]))
                        conn.commit()
                        conn.close()
                    else:
                        diaHardware.append(df[0][y])
                        diaRow.append(y+2)


                    if pd.isna(df[0][y]):
                        nhRow.append(y+2)
                
                if len(reHardware)>0 and len(reRow)>0:
                    repeat='''
                    Repeated variable:'''
                    for re in range(len(reHardware)):
                        repeat+='''
                        In Row: '''+str(reRow[re])+''' Hardware: '''+str(reHardware[re])
                    st.warning('''Repeated hardware code. Only latest diameter will be upload
                    '''+repeat)
                
                if len(diaRow)>0 and len(diaHardware)>0:
                    invalid='''
                    Ignored variable:'''
                    for re in range(len(diaHardware)):
                        invalid+='''
                        In Row: '''+str(diaRow[re])+''', Hardware: '''+str(diaHardware[re])
                    st.warning('''In Diameter column only can only accept number more than 0. Invalid variable in Diameter column will be ignored. 
                    '''+invalid)
                
                if len(nhRow)>0:
                    invalid='''
                    Row with empty hardware variable:'''
                    for re in range(len(nhRow)):
                        invalid+='''
                        In Row: '''+str(nhRow[re])
                    st.warning('''Empty value of hardware are not allowed to upload. 
                    '''+invalid)

                if len(reHardware)==0 and len(reRow)==0 and len(diaRow)==0 and len(diaHardware)==0 and len(nhRow)==0:
                    st.success("Upload successful!")
            else:
                st.warning("Wrong file uploaded!!!")


    st.sidebar.title("Add hardware diameter")
    Hardware = st.sidebar.text_input("Hardware:")
    Diameter = st.sidebar.number_input("Diameter:",min_value=0.01,step=0.01,format="%.2f")
    Upper_tolerance = st.sidebar.number_input("Upper tolerance:",min_value=0.00,step=0.01,format="%.2f")
    Lower_tolerance = st.sidebar.number_input("Lower tolerance:",min_value=0.00,step=0.01,format="%.2f")

    if st.sidebar.button("Add"):
        if (str(Hardware) and not str(Hardware).isspace()) and (str(Diameter) and not str(Diameter).isspace()) and (str(Lower_tolerance) and not str(Lower_tolerance).isspace()) and (str(Upper_tolerance) and not str(Upper_tolerance).isspace()):
            reHardware=[]
            conn = sqlite3.connect("data.db", check_same_thread=False)
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS hardware_diameter(hardware_id TEXT PRIMARY KEY, diameter NUMERIC, upper_tolerance NUMERIC, lower_tolerance NUMERIC);""")
            cur.execute("SELECT * FROM hardware_diameter WHERE (hardware_id = '%s') " %(Hardware))
            result=cur.fetchall()
            if len(result)>=1:
                cur.execute("UPDATE hardware_diameter SET diameter = ?, upper_tolerance = ?, lower_tolerance = ? WHERE hardware_id = ?", (Diameter,Upper_tolerance,Lower_tolerance,Hardware))
                reHardware.append(Hardware)
            else:
                cur.execute("INSERT INTO hardware_diameter(hardware_id,diameter,upper_tolerance,lower_tolerance) VALUES (?,?,?,?)", (Hardware,Diameter,Upper_tolerance,Lower_tolerance))

            conn.commit()
            conn.close()

            if len(reHardware)>0:
                for re in range(len(reHardware)):
                    st.warning("Repeated hardware code: "+reHardware[re]+" detected. Only latest diameter will be upload")
            else:
                st.success("Upload successful!")
        else:
            st.error("Fail too add data!!! Empty value cannot upload.")

    conn = sqlite3.connect("data.db", check_same_thread=False)
    cur = conn.cursor()
    cur.execute("SELECT * FROM hardware_diameter")
    result=cur.fetchall()
    conn.commit()
    conn.close()
    if len(result)>=1:
        table=pd.DataFrame(result)
        diametertable(table)
    else:
        st.write("NO DATA THERE!!!")

def diametertable(a):
    placeholder = st.empty()
    a.columns=["Hardware","Diameter","Upper tolerance","Lower tolerance"]

    gd = GridOptionsBuilder.from_dataframe(a)
    gd.configure_pagination(enabled=True)
    gd.configure_default_column(groupable=True,editable=True)

    gd.configure_selection(selection_mode='multiple', use_checkbox=True)
    gridoptions = gd.build()

    col6, col7 =st.columns([1,0.3])

    grid_table = AgGrid(a,gridOptions=gridoptions,height=774,width='100%',udate_mode= GridUpdateMode.MODEL_CHANGED,fit_columns_on_grid_load=True,allow_unsafe_jscode=True)

    sel_row=grid_table["selected_rows"]
    nHardware=[]
    nDiameter=[]
    nHighest=[]
    nLowest=[]
    nRow=[]
    with col2:
        if st.button("Delete"):
            for selectednum in range(len(sel_row)):
                conn = sqlite3.connect("data.db", check_same_thread=False)
                cur = conn.cursor()
                cur.execute("DELETE FROM hardware_diameter WHERE (hardware_id = '%s' AND diameter = '%s' AND upper_tolerance = '%s' AND lower_tolerance = '%s') " %(sel_row[selectednum]['Hardware'],sel_row[selectednum]['Diameter'],sel_row[selectednum]['Upper tolerance'],sel_row[selectednum]['Lower tolerance']))
                conn.commit()
                conn.close()
            st.experimental_rerun()

    with col3:
        if st.button("Update",key="1"):
            conn = sqlite3.connect("data.db", check_same_thread=False)
            cur = conn.cursor()
            cur.execute("SELECT * FROM hardware_diameter ")
            result=cur.fetchall()
            conn.commit()
            conn.close()

            for le in range(len(result)):
                if grid_table['data']['Hardware'][le]!="" and not pd.isna(grid_table['data']['Diameter'][le]) and isfloat(grid_table['data']['Diameter'][le]) and float(grid_table['data']['Diameter'][le])>0 and not pd.isna(grid_table['data']['Upper tolerance'][le]) and isfloat(grid_table['data']['Upper tolerance'][le]) and float(grid_table['data']['Upper tolerance'][le])>=0 and not pd.isna(grid_table['data']['Lower tolerance'][le]) and isfloat(grid_table['data']['Lower tolerance'][le]) and float(grid_table['data']['Lower tolerance'][le])>=0:
                    if (result[le][0]!=grid_table['data']['Hardware'][le]) or (result[le][1]!=grid_table['data']['Diameter'][le]) or (result[le][1]!=grid_table['data']['Upper tolerance'][le]) or (result[le][1]!=grid_table['data']['Lower tolerance'][le]):
                        conn = sqlite3.connect("data.db", check_same_thread=False)
                        cur = conn.cursor()
                        cur.execute("SELECT * FROM hardware_diameter WHERE (hardware_id = '%s') " %(grid_table['data']['Hardware'][le]))
                        re=cur.fetchall()
                        
                        if len(re)>=1:
                            for r in range(len(re)):
                                if result[le][0]==re[r][0]:
                                    cur.execute("UPDATE hardware_diameter SET hardware_id = ?, diameter = ?, upper_tolerance = ?, lower_tolerance = ? WHERE hardware_id = ?", (grid_table['data']['Hardware'][le],float(grid_table['data']['Diameter'][le]),float(grid_table['data']['Upper tolerance'][le]),float(grid_table['data']['Lower tolerance'][le]),grid_table['data']['Hardware'][le]))
                                else:
                                    cur.execute("DELETE FROM hardware_diameter WHERE hardware_id = '%s' " %(re[r][0]))
                                    cur.execute("UPDATE hardware_diameter SET hardware_id= ?, diameter = ?, upper_tolerance = ?, lower_tolerance = ? WHERE hardware_id = ?", (grid_table['data']['Hardware'][le],float(grid_table['data']['Diameter'][le]),float(grid_table['data']['Upper tolerance'][le]),float(grid_table['data']['Lower tolerance'][le]),result[le][0]))
                                    
                        else:
                            cur.execute("UPDATE hardware_diameter SET hardware_id = ?, diameter = ?, upper_tolerance = ?, lower_tolerance = ? WHERE hardware_id = ?", (grid_table['data']['Hardware'][le],float(grid_table['data']['Diameter'][le]),float(grid_table['data']['Upper tolerance'][le]),float(grid_table['data']['Lower tolerance'][le]),grid_table['data']['Hardware'][le]))

                        conn.commit()
                        conn.close()
                else: 
                    nHardware.append(result[le][0])
                    nDiameter.append(result[le][1])
                    nHighest.append(result[le][1])
                    nLowest.append(result[le][1])
                    nRow.append(le+1)
    with col6:
        if len(nRow)>0 and len(nHardware)>0 and len(nDiameter)>0 and len(nHighest)>0 and len(nLowest)>0:
            invalid='''
            Will remain the same:'''
            for re in range(len(nRow)):
                invalid+='''
                In Row: '''+str(nRow[re])+''', Hardware: '''+str(nHardware[re]) +''', Diameter: '''+str(nDiameter[re])+''', Upper tolerance: '''+str(nHighest[re]) +''', Lower tolerance: '''+str(nLowest[re])
            st.warning('''Empty or invalid value cannot be update 
            '''+invalid)

def user():
    st.sidebar.title("Upload User(username, role, password)")
    uploaded_file = st.sidebar.file_uploader("Choose your files", type=['xlsx', 'xls'])
    
    if st.sidebar.button("Upload data"):
        if uploaded_file is None:
            st.error("Please upload a document")
        else:
            reUsername=[]
            reRow=[]
            wrRow=[]
            nuRow=[]
            nrRow=[]
            npRow=[]
            df=pd.read_excel(uploaded_file, header=None, skiprows=[0])
            col_num=checkfile(df)
            if col_num==3:
                for y in range(len(df[0])):
                    if not pd.isna(df[0][y]) and not pd.isna(df[1][y]) and not pd.isna(df[2][y]):
                        if str(df[1][y])!="User":
                            wrRow.append(y+2)
                        else:
                            conn = sqlite3.connect("data.db", check_same_thread=False)
                            cur = conn.cursor()
                            cur.execute("""CREATE TABLE IF NOT EXISTS users(user_id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, username TEXT, password TEXT);""")
                            cur.execute("SELECT * FROM users WHERE (username = '%s') " %(df[0][y]))
                            result=cur.fetchall()
                            if len(result)>=1:
                                reUsername.append(df[0][y])
                                reRow.append(y+2)
                            else:
                                pw = [df[2][y], "def456"]
                                hashed_passwords = stauth.Hasher(pw).generate()
                                cur.execute("INSERT INTO users(role,username,password) VALUES (?,?,?)", (df[1][y],df[0][y],hashed_passwords[0]))
                            conn.commit()
                            conn.close()

                    if pd.isna(df[0][y]):
                        nuRow.append(y+2)

                    if pd.isna(df[1][y]):
                        nrRow.append(y+2)

                    if pd.isna(df[2][y]):
                        npRow.append(y+2)

                if len(reUsername)>0 and len(reRow)>0:
                    repeat='''
                    Repeated username:'''
                    for re in range(len(reUsername)):
                        repeat+='''
                        In row: '''+str(reRow[re])+''' Username: '''+str(reUsername[re])
                    st.warning('''Please change another username and upload again
                    '''+repeat)
                
                if len(nuRow)>0:
                    invalid='''
                    Row with empty Username:'''
                    for re in range(len(nuRow)):
                        invalid+='''
                        In Row: '''+str(nuRow[re])
                    st.warning('''Empty value of username are not allowed to upload. 
                    '''+invalid)

                if len(nrRow)>0:
                    invalid='''
                    Row with empty Role:'''
                    for re in range(len(nrRow)):
                        invalid+='''
                        In Row: '''+str(nrRow[re])
                    st.warning('''Empty value of role are not allowed to upload. 
                    '''+invalid)
            
                if len(npRow)>0:
                    invalid='''
                    Row with empty Password:'''
                    for re in range(len(npRow)):
                        invalid+='''
                        In Row: '''+str(npRow[re])
                    st.warning('''Empty value of password are not allowed to upload. 
                    '''+invalid)

                if len(wrRow)>0:
                    invalid='''
                    Invalid Role variable:'''
                    for re in range(len(wrRow)):
                        invalid+='''
                        In Row: '''+str(wrRow[re])
                    st.warning('''Role can only be Admin or User, invalid role are not allowed. 
                    '''+invalid)
                

                if len(reUsername)==0 and len(reRow)==0 and len(nuRow)==0 and len(nrRow)==0 and len(npRow)==0 and len(wrRow)==0:
                    st.success("Upload successful!")
            else:
                st.warning("Wrong file uploaded!!!")



    st.sidebar.title("Add User")
    Username = st.sidebar.text_input("Username:")
    Role = st.sidebar.selectbox("Role:",('','Admin', 'User'))
    Password = st.sidebar.text_input("Password:")
    pw = [Password, "def456"]
    hashed_passwords = stauth.Hasher(pw).generate()
    if st.sidebar.button("Add"):
        if (Username and not Username.isspace()) and (Role and not Role.isspace()) and (Password and not Password.isspace()):
            reUsername=[]
            conn = sqlite3.connect("data.db", check_same_thread=False)
            cur = conn.cursor()
            cur.execute("""CREATE TABLE IF NOT EXISTS users(user_id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, username TEXT, password TEXT);""")
            cur.execute("SELECT * FROM users WHERE (username = '%s') " %(Username))
            result=cur.fetchall()
            if len(result)>=1:
                cur.execute("UPDATE users SET role = ? WHERE username = ?", (Role,Username))
                reUsername.append(Username)
            else:
                cur.execute("INSERT INTO users(role,username,password) VALUES (?,?,?)", (Role,Username,hashed_passwords[0]))

            conn.commit()
            conn.close()

            if len(reUsername)>0:
                for re in range(len(reUsername)):
                    st.warning("Repeated Username: "+reUsername[re]+" detected. Only one username will be upload")
            else:
                st.success("Upload successful!")
        else:
            st.error("Fail too add data!!! Empty value cannot upload.")

    conn = sqlite3.connect("data.db", check_same_thread=False)
    cur = conn.cursor()
    cur.execute("SELECT * FROM users")
    result=cur.fetchall()
    conn.commit()
    conn.close()
    if len(result)>=1:
        for de in range(len(result)):
            mylist=list(result[de])
            mylist.remove(result[de][0])
            mylist.remove(result[de][3])
            result[de]=tuple(mylist)
        table=pd.DataFrame(result)
        usertable(table)
    else:
        st.write("NO DATA THERE!!!")

def usertable(a):
    placeholder = st.empty()
    a.columns=["Role","Username"]

    gd = GridOptionsBuilder.from_dataframe(a)
    gd.configure_pagination(enabled=True)
    gd.configure_default_column(groupable=True,editable=True)

    gd.configure_selection(selection_mode='multiple', use_checkbox=True)
    gridoptions = gd.build()

    col6, col7 =st.columns([1,0.3])


    grid_table = AgGrid(a,gridOptions=gridoptions,height=350,width='100%',udate_mode= GridUpdateMode.MODEL_CHANGED,fit_columns_on_grid_load=True,allow_unsafe_jscode=True)

    sel_row=grid_table["selected_rows"]
    nRole=[]
    nUser=[]
    nRow=[]
    with col2:
        if st.button("Delete"):
            for selectednum in range(len(sel_row)):
                conn = sqlite3.connect("data.db", check_same_thread=False)
                cur = conn.cursor()
                cur.execute("DELETE FROM users WHERE (username = '%s') " %(sel_row[selectednum]['Username']))
                conn.commit()
                conn.close()
            st.experimental_rerun()

    with col3:
        if st.button("Update",key="1"):
            conn = sqlite3.connect("data.db", check_same_thread=False)
            cur = conn.cursor()
            cur.execute("SELECT * FROM users ")
            result=cur.fetchall()
            conn.commit()
            conn.close()
            for le in range(len(result)):
                if (grid_table['data']['Role'][le]=="Admin" or grid_table['data']['Role'][le]=="User") and not grid_table['data']['Username'][le].isspace() and grid_table['data']['Username'][le]!="" :
                    if (result[le][1]!=grid_table['data']['Role'][le]) or (result[le][2]!=grid_table['data']['Username'][le]):
                        conn = sqlite3.connect("data.db", check_same_thread=False)
                        cur = conn.cursor()
                        cur.execute("SELECT * FROM users WHERE (username = '%s') " %(grid_table['data']['Username'][le]))
                        re=cur.fetchall()
                        
                        if len(re)>=1:
                            for r in range(len(re)):
                                if result[le][0]==re[r][0]:
                                    cur.execute("UPDATE users SET role = ?, username = ? WHERE user_id = ?", (grid_table['data']['Role'][le],grid_table['data']['Username'][le],result[le][0]))
                                else:
                                    cur.execute("DELETE FROM users WHERE user_id = '%s' " %(re[r][0]))
                                    cur.execute("UPDATE users SET role= ?, username = ? WHERE user_id = ?", (grid_table['data']['Role'][le],grid_table['data']['Username'][le],result[le][0]))
                                    
                        else:
                            cur.execute("UPDATE users SET role = ?, username = ? WHERE user_id = ?", (grid_table['data']['Role'][le],grid_table['data']['Username'][le],result[le][0]))

                        conn.commit()
                        conn.close()
                else: 
                    nRole.append(result[le][1])
                    nUser.append(result[le][2])
                    nRow.append(le+1)  
    with col6:
        if len(nRole)>0 and len(nRow)>0 and len(nUser)>0:
            invalid='''
            Will remain the same:'''
            for re in range(len(nRole)):
                invalid+='''
                In Row: '''+str(nRow[re])+''' Role: '''+str(nRole[re])+''', Username: '''+str(nUser[re])
            st.warning('''Empty or invalid value cannot be update 
            '''+invalid)

def rpassword():
    st.title("Reset Password")
    nPassword = st.text_input("New Password:")
    rPassword = st.text_input("Confirm Password:")
 
    if st.button("Reset"):
        if nPassword==rPassword:
            npw = [nPassword, "def456"]
            nhashed_passwords = stauth.Hasher(npw).generate()
            conn = sqlite3.connect("data.db", check_same_thread=False)
            cur = conn.cursor()
            cur.execute("SELECT * FROM users WHERE (username = '%s') " %(st.session_state["username"]))
            result=cur.fetchall()
            cur.execute("UPDATE users SET password= ? WHERE user_id = ?", (nhashed_passwords[0],result[0][0]))
            conn.commit()
            conn.close()
            st.success("Password Rest")
        else:
            st.warning("New password and confirm password not match")
        
def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

def checkfile(file):
    try:
        for i in range(5):
            a=file[i][0]
        return i
    except :
        return i

# --- USER AUTHENTICATION ---
role=[]
usernames=[]
password=[]

cur.execute("SELECT * FROM users")
result=cur.fetchall()

for row in result:
    role.append(row[1])
    usernames.append(row[2])
    password.append(row[3])

conn.commit()
conn.close()

authenticator = stauth.Authenticate(role, usernames, password,
    "fyp_dashboard", "abcdef", cookie_expiry_days=30)

role, authentication_status, username = authenticator.login("Login", "main")

if st.session_state['authentication_status'] == False:
    st.error("Username/password is incorrect")

if st.session_state['authentication_status'] == None:
    st.warning("Please enter your username and password")


if st.session_state['authentication_status'] and st.session_state["name"]=="Admin":
        
    col4, col5 =st.columns([1.5,0.3])
    selected = option_menu(
        menu_title=None,  # required
        options=["Drawing Recognition", "BOM","Diameter Spec","Recognition History","User","Reset Password"],  # required
        menu_icon="cast", #optional
        default_index=0,
        orientation="horizontal",
        styles={
                "container": {"padding": "0!important", "background-color": "#BEBEBE"},
                "icon": {"visibility":"hidden"},
                "nav-link": {
                    "font-size": "17px",
                    "text-align": "center",
                    "margin": "0px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        )

    col1,col2,col3=st.columns([1.4,0.1,0.1])

    with col5:
        authenticator.logout("Logout", "main")
            
    if selected =="Drawing Recognition":
        main()

    if selected =="BOM":
        BOM()

    if selected =="Diameter Spec":
        diameter()

    if selected =="Recognition History":
        history()
    
    if selected =="User":
        user()

    if selected =="Reset Password":
        rpassword()

elif st.session_state['authentication_status']:

    col4, col5 = st.columns([1.5, 0.3])
    selected = option_menu(
        menu_title=None,  # required
        options=["Drawing Recognition", "Recognition History", "Reset Password"],  # required
        menu_icon="cast",  # optional
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#BEBEBE"},
            "icon": {"visibility": "hidden"},
            "nav-link": {
                "font-size": "20px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "green"},
        },
    )

    col1, col2, col3 = st.columns([1.4, 0.1, 0.1])

    with col5:
        authenticator.logout("Logout", "main")

    if selected == "Drawing Recognition":
        main()

    if selected == "Recognition History":
        history()

    if selected == "Reset Password":
        rpassword()

