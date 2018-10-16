import os

import click as click

from audionmf.formats.audio_file import AudioFile
from audionmf.formats.audio_file_compressed import AudioFileCompressed


def get_filename_ext(path):
    return os.path.splitext(os.path.basename(path))


def get_output_handle(input_filename, filetype):
    raw_name = get_filename_ext(input_filename)[0]
    target_name = raw_name + '.' + filetype
    return open(target_name, 'wb')


@click.group()
def cli():
    pass


@cli.command()
@click.argument('input_file', type=click.File('rb'))
@click.argument('output_file', type=click.File('wb'), required=False)
def compress(input_file, output_file):
    filename = input_file.name
    filetype = get_filename_ext(filename)[1].lower()[1:]
    if output_file is None:
        output_file = get_output_handle(filename, 'anmf')

    audio = AudioFile.read_file(input_file, filetype)

    if audio is None:
        print('invalid file format: {}'.format(filetype))
    else:
        audio.compress(output_file)

    input_file.close()
    output_file.close()


@cli.command()
@click.argument('input_file', type=click.File('rb'))
@click.argument('output_file', type=click.File('wb'), required=False)
@click.option('-t', '--filetype', type=click.Choice(['wav', 'flac']), default='wav')
def decompress(input_file, output_file, filetype):
    if output_file is None:
        output_file = get_output_handle(input_file.name, filetype)

    audio = AudioFileCompressed.read_file(input_file)
    audio.decompress(output_file, filetype)

    input_file.close()
    output_file.close()
