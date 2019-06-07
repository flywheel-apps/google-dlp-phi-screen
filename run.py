#!/usr/bin/env python

import os
import requests
import json
import logging
from io import BytesIO
import base64
import PIL
import zipfile
import tempfile
import numpy as np
import pandas as pd
import pydicom
from PIL import ImageDraw
# Set up logging
logging.basicConfig()
log = logging.getLogger('google-dlp-gear')


def dlp_request_handler(project_string, google_api_key, dlp_endpoint_str, request_data, retry_count=5):
    endpoint = 'https://dlp.googleapis.com/v2/projects/{}/{}?key={}'.format(
        project_string,
        dlp_endpoint_str,
        google_api_key)
    try:
        response = requests.post(endpoint, data=request_data, timeout=10)
        if response.ok:
            response_json = json.loads(response.text)
            return response_json
        else:
            log.error('Invalid response: {}'.format(response.text))
            log.error('Exiting...')
            os.sys.exit(1)
    except requests.exceptions.RequestException as e:
        if retry_count > 0:
            log.warning('HTTP Request exception: {}'.format(e))
            log.warning('Retrying...')
            retry_count -= 1
            dlp_request_handler(project_string, google_api_key, dlp_endpoint_str, request_data, retry_count)
        else:
            log.warning('HTTP Request exception: {}'.format(e))
            log.error('Retries exceeded. Exiting...')
            os.sys.exit(1)


def get_valid_info_types(google_api_key, retry_count=5):
    # Get a list of valid infoTypes
    valid_info_types = list()
    try:
        # Make request to api for infoTypes
        infotype_response = requests.get('https://dlp.googleapis.com/v2/infoTypes?key={}'.format(google_api_key),
                                         timeout=10)
        # Check if response is ok
        if infotype_response.ok:
            # Load text from response to python object
            valid_info_type_dict = json.loads(infotype_response.text)['infoTypes']
            for info_type_dict in valid_info_type_dict:
                valid_info_types.append(info_type_dict['name'])
            return valid_info_types
        else:
            log.error('Unable to retrieve infoTypes from Google. Response: {}'.format(infotype_response.text))
    except requests.exceptions.RequestException as e:
        if retry_count > 0:
            log.warning('HTTP Request exception: {}'.format(e))
            log.warning('Retrying...')
            retry_count -= 1
            get_valid_info_types(google_api_key, retry_count)
        else:
            log.warning('HTTP Request exception: {}'.format(e))
            log.error('Retries exceeded. Exiting...')
            os.sys.exit(1)


def generate_inspect_config(info_type_list, min_likelihood, include_quote):
    inspect_config = dict()

    # Add minLikelihood
    inspect_config['minLikelihood'] = min_likelihood

    # Add includeQuote
    inspect_config['includeQuote'] = include_quote

    # Create infoTypes object
    info_types_object = list()
    # If valid infoTypes were provided, generate an object for them
    if info_type_list:
        # Convert list of infoTypes to list of dicts
        for info_type in info_type_list:
            info_type_dict = dict()
            info_type_dict['name'] = info_type
            info_types_object.append(info_type_dict)
    # If no valid infoTypes were provided, set to "ALL_BASIC"
    else:
        info_type_dict = dict()
        info_type_dict['name'] = "ALL_BASIC"
        info_types_object.append(info_type_dict)

    # Add infoTypes
    inspect_config['infoTypes'] = info_types_object

    return inspect_config


def inspect_pixel_array(input_array, inspect_config, project, key):
    # Make a copy of the input array
    image_data = input_array.copy()
    # Convert to float 64 for scaling to 255
    image_data = image_data.astype('float64')
    # scale to 255 for PNG export
    image_data *= (255.0 / image_data.max())
    # convert to PIL object for image operations
    image = PIL.Image.fromarray(np.uint8(image_data))
    # Convert image to a base64-encoded string
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    b64_img_string = str(base64.b64encode(buffered.getvalue()))[2:-1]
    # Prepare image inspection object
    image_inspect_object = {
        "item": {
            "byteItem": {
                "data": b64_img_string,
                "type": "IMAGE_PNG"
            }
        },
        "inspectConfig": inspect_config
    }

    image_inspect_data = json.dumps(image_inspect_object)
    image_inspect_response = dlp_request_handler(project, key, 'content:inspect', image_inspect_data)
    return image_inspect_response


def inspect_text(inspect_config, text_object, project, key):
    inspect_text_object = dict()
    inspect_text_object['item'] = text_object
    inspect_text_object['inspectConfig'] = inspect_config
    image_inspect_data = json.dumps(inspect_text_object)
    inspection_response = dlp_request_handler(project, key, 'content:inspect', image_inspect_data)
    return inspection_response


def extract_return_path(input_filepath):
    if zipfile.is_zipfile(input_filepath):
        # Make a temporary directory
        temp_dirpath = tempfile.mkdtemp()
        # Extract to this temporary directory
        with zipfile.ZipFile(input_filepath) as zip_object:
            zip_object.extractall(temp_dirpath)
            # Return the list of paths
            file_list = zip_object.namelist()
            # Get full paths and remove directories from list
            file_list = [os.path.join(temp_dirpath, file) for file in file_list if not file.endswith('/')]
    else:
        log.info('Not a zip. Attempting to read %s directly' % os.path.basename(input_filepath))
        file_list = [input_filepath]
    return file_list


def import_dicom_header_as_dict(dcm_filepath):
    header_dict = dict()
    include_vr = ['PN', 'CS', 'AS', 'UT', 'UN', 'LT', 'ST', 'DA']
    header_dict['file_name'] = os.path.basename(dcm_filepath)
    try:
        dataset = pydicom.read_file(dcm_filepath)
    except pydicom.filereader.InvalidDicomError:
        log.warning('Invalid DICOM file: {}'.format(dcm_filepath))
        return None
    for element in dataset:
        if (element.VR in include_vr) and element.keyword:
            key = element.keyword
            value = element.value
            header_dict[key] = str(value)
    return header_dict


def dicom_list_to_data_frame(dicom_file_list):
    df_list = list()
    for dicom_file in dicom_file_list:
        tmp_dict = import_dicom_header_as_dict(dicom_file)
        if tmp_dict:
            for key in tmp_dict:
                if type(tmp_dict[key]) == list:
                    tmp_dict[key] = str(tmp_dict[key])
                else:
                    tmp_dict[key] = [tmp_dict[key]]
            df_tmp = pd.DataFrame.from_dict(tmp_dict)
            df_list.append(df_tmp)
        else:
            log.info('Returned empty dict for file: {}'.format(dicom_file))
            log.info('Trying next file..')
            continue
    if df_list:
        df = pd.concat(df_list, ignore_index=True, sort=True)
        return df
    else:
        log.error('Unable to parse any DICOMS')
        os.sys.exit(1)


def df_to_table(input_df):
    table_headers_list = list()
    table_rows_list = list()
    headers = input_df.columns
    for column in headers:
        header_dict = {"name": column}
        table_headers_list.append(header_dict)
    for index, row in input_df.iterrows():
        row_list = list()
        header_row = zip(headers, row.values)
        for key, value in header_row:
            value = '{}: \'{}\''.format(key,value)
            value_dict = {'string_value': value}
            row_list.append(value_dict)
        row_dict = {'values': row_list}
        table_rows_list.append(row_dict)
    table = dict()
    table['headers'] = table_headers_list
    table['rows'] = table_rows_list
    table = {'table': table}
    return table


def format_response_findings(response, df=None, file_name=None):
    if response['result']:
        finding_list = list()
        for finding in response['result']['findings']:
            finding_dict = dict()
            finding_dict['infoType'] = finding['infoType']
            finding_dict['quote'] = finding['quote']
            finding_dict['likelihood'] = finding['likelihood']
            if 'imageLocation' in finding['location']['contentLocations'][0].keys():
                finding_dict['inspection_type'] = 'dicom image'
                finding_dict['boundingBoxes'] = finding['location']['contentLocations'][0]['imageLocation']['boundingBoxes']
                if file_name:
                    finding_dict['file_name'] = file_name
            else:
                finding_dict['inspection_type'] = 'dicom header'
                finding_dict['dicom_field'] = finding['location']['contentLocations'][0]['recordLocation']['fieldId']['name']
                table_location = finding['location']['contentLocations'][0]['recordLocation']['tableLocation']
                if isinstance(df, pd.core.frame.DataFrame):
                    df_row = int(table_location['rowIndex'])
                    file_name = df.loc[df_row, 'file_name']
                    finding_dict['file_name'] = file_name
                else:
                    finding_dict['file_name'] = 'all_files'
            finding_list.append(finding_dict)
        return finding_list
    else:
        return []


def get_redact_coords(findings):
    coord_list = []
    for finding in findings:
        bbox_list = finding['boundingBoxes']
        for bbox_dict in bbox_list:
            x0 = bbox_dict['left']
            y0 = bbox_dict['top']
            x1 = x0 + bbox_dict['width']
            y1 = y0 + bbox_dict['height']
            coords = [x0, y0, x1, y1]
            coord_list.append(coords)
    return coord_list


def redact_image(input_image, coordinate_list):
    output_image = input_image.copy()
    draw = PIL.ImageDraw.Draw(output_image)
    for coords in coordinate_list:
        draw.rectangle(coords, fill=0)
    return output_image


def inspect_and_redact_dicom_images(dicom_file_list, inspect_config, project, key, redact=False, dicom_tag_list=[]):
    image_inspect_findings = list()
    for dicom_filepath in dicom_file_list:
        try:
            dataset = pydicom.read_file(dicom_filepath)
        except pydicom.filereader.InvalidDicomError:
            log.warning('Invalid DICOM file: {}'.format(dicom_filepath))
            log.warning('Trying next file')
            continue
        pixel_data = dataset.pixel_array
        image_inspect_response = inspect_pixel_array(pixel_data, inspect_config, project, key)
        dicom_file_name = os.path.basename(dicom_filepath)
        image_findings = format_response_findings(image_inspect_response, file_name=dicom_file_name)
        if image_findings:
            image_inspect_findings.append(image_findings)
            if redact:
                redact_coords = get_redact_coords(image_findings)
                dicom_image = PIL.Image.fromarray(dataset.pixel_array)
                redacted_image = redact_image(dicom_image, redact_coords)
                for tag in dicom_tag_list:
                    dataset.data_element(tag).value = ''
                dataset.PixelData = np.asarray(redacted_image).astype('int16').tobytes()
                pydicom.filewriter.write_file(dicom_filepath, dataset, write_like_original=True)

    return image_inspect_findings


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), file)


if __name__ == '__main__':

    # Gear basics
    input_folder = '/flywheel/v0/input/file/'
    output_folder = '/flywheel/v0/output/'

    # Declare config file path
    config_file_path = '/flywheel/v0/config.json'

    # Load config file
    with open(config_file_path) as config_data:
        config = json.load(config_data)

    # Load Google API information
    google_api_filepath = config['inputs']['google_api_key_file']['location']['path']
    with open(google_api_filepath) as api_data:
        api_info = json.load(api_data)

    # Load config options
    config_options = config['config']
    # Determine redact value
    redact = config_options['redact']
    log.info("Getting info types")
    # Get infoTypes from config
    input_info_types = list()
    for key in config_options:
        if key.startswith("infoType"):
            input_info_types.append(config_options[key])
    # Query the DLP API for valid infoType list
    valid_info_type_list = get_valid_info_types(api_info['api_key'])
    # Only return unique values that match the infoTypes returned by the API
    valid_input_info_type_list = list(set(input_info_types) & set(valid_info_type_list))
    log.info('Valid infoTypes: {}'.format(str(valid_info_type_list)))
    log.info("Creating inspection config")
    # Create inspectConfig
    inspect_config_object = generate_inspect_config(valid_input_info_type_list,
                                                    config_options['minLikelihood'],
                                                    config_options['includeQuote'])

    response_result_list = list()
    # Set dicom filepath
    dicom_filepath = config['inputs']['input_dicom']['location']['path']

    log.info("Extracting DICOMS")
    # extract dicoms and return list of paths
    extracted_file_list = extract_return_path(dicom_filepath)

    # Import DICOM header information as a dataframe
    dicom_df = dicom_list_to_data_frame(extracted_file_list)

    # Many tags in DICOM headers are static for an archive, so we're going to be thrifty
    not_unique_columns = dicom_df.loc[:, dicom_df.apply(lambda x: x.nunique()) == 1].columns
    unique_columns = dicom_df.loc[:, dicom_df.apply(lambda x: x.nunique()) > 1].columns
    # We need to convert the data to Google's table format for the requests
    unique_obj = df_to_table(dicom_df.loc[:, unique_columns])
    # For columns where all values are the same, we only need one row
    not_unique_obj = df_to_table(dicom_df.loc[0:0, not_unique_columns])

    log.info('Inspecting header data for {} DICOM files...'.format(len(dicom_df)))

    # Submit the DLP requests for header data
    unique_response = inspect_text(inspect_config_object, unique_obj, api_info['project'], api_info['api_key'])
    not_unique_response = inspect_text(inspect_config_object, not_unique_obj, api_info['project'], api_info['api_key'])

    # Format objects for output JSON and redaction
    unique_findings = format_response_findings(unique_response, dicom_df)
    not_unique_findings = format_response_findings(not_unique_response)
    header_findings = unique_findings + not_unique_findings

    redact_tags = [finding['dicom_field'] for finding in header_findings]
    image_findings = inspect_and_redact_dicom_images(extracted_file_list,
                                                     inspect_config_object,
                                                     api_info['project'],
                                                     api_info['api_key'],
                                                     redact=redact,
                                                     dicom_tag_list=redact_tags)


    all_findings = header_findings + image_findings
    result_filepath = os.path.join(output_folder, 'phi.findings.json')
    metadata_out_filepath = os.path.join(output_folder, '.metadata.json')
    if all_findings:
        log.info('Writing results...')
        metadata_out = {
            'acquisition': {
                'tags': ['PHI']
            },
            'session': {
                'tags': ['PHI']
            }
        }
        if redact and image_findings:
            extracted_directory = os.path.dirname(extracted_file_list[0])
            redacted_dicom_path = output_folder + "redacted_" + os.path.basename(dicom_filepath)
            log.info('Saving redacted DICOM files to {}'.format(redacted_dicom_path))
            zipf = zipfile.ZipFile(redacted_dicom_path, 'w', zipfile.ZIP_DEFLATED)
            zipdir(extracted_directory, zipf)
            zipf.close()
            metadata_out['acquisition']['files'] = [
                {'type': 'dicom', 'name': str(os.path.basename(redacted_dicom_path))}
            ]
        with open(metadata_out_filepath, 'w') as outfile:
            json.dump(metadata_out, outfile, separators=(', ', ': '), sort_keys=True, indent=4)
        with open(result_filepath, 'w') as outfile:
            json.dump(all_findings, outfile, separators=(', ', ': '), sort_keys=True, indent=4)
    os.sys.exit(0)
