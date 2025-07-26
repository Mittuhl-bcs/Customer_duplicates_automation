import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import json
import zipfile
import os
from datetime import datetime


def zip_excel_files(file_paths, zip_filename):
    # file_paths: list of Excel file paths
    zip_path = zip_filename + '.zip'
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for file_path in file_paths:
            arcname = os.path.basename(file_path)  # just the filename inside the zip
            zipf.write(file_path, arcname=arcname)
    return zip_path

def send_email(attachment_filename, attachment_display_name):
    with open("D:\\Item_replenishment_report_automation\\Credentials.json", "r+") as crednt:
        data = json.load(crednt)
        password = data["password"]

    try:
        sender_email = "Bcs.notifications@building-controls.com"
        sender_password = password
        receiver_emails = ["mithul.murugaadev@building-controls.com",
                           "brian.ackerman@building-controls.com",
            "adam.martinez@building-controls.com",
            "harriette.henderson@building-controls.com",
            "jason.bail@building-controls.com"]
        subject = 'Customer data (P21) duplicates'
        body = """Hi Team,
        
A report of new customer data duplicates are created and is shared through this mail. The duplicates are identified using the whole customers data. 

Please find the file attached.

Regards,
Mithul
                """

        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = ', '.join(receiver_emails)
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))

        with open(attachment_filename, 'rb') as attachment:
            part = MIMEBase('application', 'octet-stream')
            part.set_payload(attachment.read())

        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename="{attachment_display_name}"')
        message.attach(part)
        text = message.as_string()

        server = smtplib.SMTP('smtp.office365.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.sendmail(sender_email, receiver_emails, text)
        server.quit()

        return True

    except Exception as e:
        raise ValueError(f'Failed to send email: {e}')


def sender(excel_file_paths):
    current_time = datetime.now()
    day = current_time.day
    month = current_time.strftime("%b")
    year = current_time.year

    zip_filename = "Customer_duplicates"
    zip_filepath = zip_excel_files(excel_file_paths, zip_filename)
    attachment_display_name = f"customer_duplicates_{day}_{month}_{year}.zip"
    send_email(zip_filepath, attachment_display_name)
