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


def compress(input_file, output_file, filetype):
    audio = AudioFile.read_file(input_file, filetype)

    if audio is None:
        print('invalid file format: {}'.format(filetype))
    else:
        audio.compress(output_file)


def decompress(input_file, output_file, filetype):
    audio = AudioFileCompressed.read_file(input_file)
    audio.decompress(output_file, filetype)


@click.group()
def cli():
    pass


@cli.command(name='compress')
@click.argument('input_file', type=click.File('rb'))
@click.argument('output_file', type=click.File('wb'), required=False)
def compress_command(input_file, output_file):
    filename = input_file.name
    filetype = get_filename_ext(filename)[1].lower()[1:]
    if output_file is None:
        output_file = get_output_handle(filename, 'anmf')

    compress(input_file, output_file, filetype)

    input_file.close()
    output_file.close()


@cli.command(name='decompress')
@click.argument('input_file', type=click.File('rb'))
@click.argument('output_file', type=click.File('wb'), required=False)
@click.option('-t', '--filetype', type=click.Choice(['wav', 'flac']), default='wav')
def decompress_command(input_file, output_file, filetype):
    if output_file is None:
        output_file = get_output_handle(input_file.name, filetype)

    decompress(input_file, output_file, filetype)

    input_file.close()
    output_file.close()


@cli.command(name='debug')
@click.argument('input_file', type=click.File('rb'))
def debug_command(input_file):
    with open('debug.anmf', 'wb') as anmf_file:
        compress(input_file, anmf_file, 'wav')
    with open('debug.anmf', 'rb') as anmf_file, open('debug.wav', 'wb') as wav_file:
        decompress(anmf_file, wav_file, 'wav')
