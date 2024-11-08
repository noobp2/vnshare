import os
import pandas as pd
from email.message import EmailMessage
from guerilla.core.config import config_parser

if os.name == 'nt':
    import win32com.client as emc
else:
    import smtplib as emc

def send_email_outlook(subject, body, to_address, attachment_path=None):
    
    outlook = emc.Dispatch('Outlook.Application')
    mail = outlook.CreateItem(0)  # 0 represents a mail item

    # Set email properties
    mail.Subject = subject
    mail.HTMLBody = f"<p>{body}</p>"
    mail.To = to_address

    # Attach file if specified
    if attachment_path:
        mail.Attachments.Add(attachment_path)
        

    # Send the email
    mail.Send()

def send_email_gmail(subject, body, to_address, attachment_path=None):
    # Get email sender and password from configuration file
    cfg = config_parser()
    email_sender = cfg['ems']['sender']
    email_password = cfg['ems']['pwd']
    
    em = EmailMessage()
    em['To'] = to_address
    em['From'] = email_sender
    em['Subject'] = subject
    em.set_content(body, subtype='html')

    # Attach file if specified
    if attachment_path:
        with open(attachment_path, 'rb') as f:
            file_data = f.read()
            file_name = os.path.basename(attachment_path)
        em.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)
    
    # Send the email
    with emc.SMTP('smtp.gmail.com', 587) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.login(email_sender, email_password)
        smtp.send_message(em)
    print("Email sent successfully.")

def send_email(subject, body, to_address, attachment_path=None):
    if os.name == 'nt':
        send_email_outlook(subject, body, to_address, attachment_path)
    else:
        send_email_gmail(subject, body, to_address, attachment_path)

def dataframe_to_html(data_df: pd.DataFrame):
    table_html = """
        <style>
            table {
                border-collapse: collapse;
                width: 100%;
                background-color: #ECEBEB;
            }
            th,td {
                border: 1px solid #FDFEFE;
                text-align: center;
                padding: 8px;
            }

            th {
                background-color: #4CAF50;
            }
        </style>
        """ + data_df.to_html(index=False)
    return table_html