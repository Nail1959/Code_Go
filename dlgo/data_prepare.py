from __future__ import print_function
from __future__ import absolute_import
import os
import glob
import os.path
import tarfile
import gzip
import shutil
import numpy as np
import multiprocessing
from os import sys
from keras.utils import to_categorical
import time

from dlgo.gosgf import Sgf_game
from dlgo.goboard_fast import Board, GameState, Move, Board_Ext
from dlgo.gotypes import Player, Point
from dlgo.data.index_processor import KGSIndex
from dlgo.data.sampling import Sampler
from dlgo.data.generator import DataGenerator
from dlgo.encoders.base import get_encoder_by_name
from dlgo.encoders.my_sevenplane_s import MySevenPlaneEncoder_S


def worker(jobinfo):
    try:
        clazz, encoder, zip_file, data_file_name, game_list = jobinfo
        clazz(encoder=encoder).process_zip(zip_file, data_file_name, game_list)
    except (KeyboardInterrupt, SystemExit):
        raise Exception('>>> Exiting child process.')

class GoDataProcessor:
    count_stones_debute: object

    def __init__(self, encoder='simple', data_directory='data',
                 count_stones_debute = None,
                 count_stones_middle = None,
                 count_stones_end = None):
        self.encoder_string = encoder
        self.encoder = get_encoder_by_name(encoder, 19)
        self.data_dir = data_directory
        self.count_stones_debute = count_stones_debute # Nail counts stones on board debute
        self.count_stones_middle = count_stones_middle  # Nail counts stones on board middle
        self.count_stones_end = count_stones_end  # Nail counts stones on board end

# tag::load_generator[]
    def load_go_data(self, data_type='train', num_samples=1000,
                     use_generator=False):
        index = KGSIndex(data_directory=self.data_dir)
        index.download_files()

        sampler = Sampler(data_dir=self.data_dir)
        data = sampler.draw_data(data_type, num_samples)

        self.map_to_workers(data_type, data)  # <1>
        if use_generator:
            generator = DataGenerator(self.data_dir, data)
            return generator  # <2>
        else:
            features_and_labels = self.consolidate_games(data_type, data)
            return features_and_labels  # <3>

# <1> Map workload to CPUs
# <2> Either return a Go data generator...
# <3> ... or return consolidated data as before.
# end::load_generator[]

    def unzip_data(self, zip_file_name):
        this_gz = gzip.open(self.data_dir + '/' + zip_file_name)

        tar_file = zip_file_name[0:-3]
        this_tar = open(self.data_dir + '/' + tar_file, 'wb')

        shutil.copyfileobj(this_gz, this_tar)
        this_tar.close()
        return tar_file

    def process_zip(self, zip_file_name, data_file_name, game_list):
        tar_file = self.unzip_data(zip_file_name)
        zip_file = tarfile.open(self.data_dir + '/' + tar_file)
        name_list = zip_file.getnames()
        total_examples = self.num_total_examples(zip_file, game_list, name_list)

        shape = self.encoder.shape()
        feature_shape = np.insert(shape, 0, np.asarray([total_examples]))
        features = np.zeros(feature_shape)
        labels = np.zeros((total_examples,))

        counter = 0

        board=Board(19,19)  #Nail
        board_ext = Board_Ext(board) # Nail
        for index in game_list:

            name = name_list[index + 1]
            if not name.endswith('.sgf'):
                raise ValueError(name + ' is not a valid sgf')
            sgf_content = zip_file.extractfile(name).read()
            sgf = Sgf_game.from_string(sgf_content)

            game_state, first_move_done, board_ext = self.get_handicap(sgf)
            # if first_move_done :  # Nail ignore handicap
            #      continue  # Ignore games with handicap
            if self.encoder.name()[:2] == 'my' and  first_move_done == False: # Not handicap
                board_ext = Board_Ext(game_state.board)  #inserted Nail
            for item in sgf.main_sequence_iter():
                color, move_tuple = item.get_move()
                point = None
                if color is not None:
                    if move_tuple is not None:
                        row, col = move_tuple
                        point = Point(row + 1, col + 1)
                        move = Move.play(point)

                    else:
                        move = Move.pass_turn()
                    if first_move_done and point is not None:
                        encode = True
                        # # Data only for debute Nail
                        # if self.count_stones_debute is not None and \
                        #     self.count_stones_middle is None and \
                        #     self.count_stones_end is None and \
                        #     board_ext.count_stones() > self.count_stones_debute:
                        #     encode = False
                        # # Data for middle game Nail
                        # if self.count_stones_debute is not None and \
                        #         self.count_stones_middle is not None and \
                        #         self.count_stones_end is None and \
                        #         self.board_ext.count_stones() <= self.count_stones_debute and board_ext.count_stones() > self.count_stones_middle:
                        #     encode = False
                        #
                        # # Data for end
                        # if self.count_stones_middle is not None and \
                        #         self.count_stones_end is not None and \
                        #         self.board_ext.count_stones() <= self.count_stones_middle and board_ext.count_stones() > self.count_stones_end:
                        #     encode = False
                        if encode == True: #Nail
                            if self.encoder.name()[:2] == 'my':
                                features[counter] = self.encoder.encode(game_state,board_ext)  #Nail
                            else:
                                features[counter] = self.encoder.encode(game_state)

                            labels[counter] = self.encoder.encode_point(point)
                            counter += 1

                    game_state = game_state.apply_move(move)
                    if self.encoder.name()[:2] == 'my':
                        board_ext.place_stone_ext(game_state.board, color, point)  # Inserted Nail
                    # Nail
                    first_move_done = True

        feature_file_base = self.data_dir + '/' + data_file_name + '_features_%d'
        label_file_base = self.data_dir + '/' + data_file_name + '_labels_%d'

        chunk = 0  # Due to files with large content, split up after chunksize
        chunksize = 1024
        start_time_all = time.time()

        while features.shape[0] >= chunksize:
            start_time = time.time()
            feature_file = feature_file_base % chunk
            label_file = label_file_base % chunk
            chunk += 1
            current_features, features = features[:chunksize], features[chunksize:]
            current_labels, labels = labels[:chunksize], labels[chunksize:]
            np.save(feature_file, current_features)
            np.save(label_file, current_labels)
            # Inserted Nail
            print("Chunk = ", chunk," File for training Current_features: ", feature_file)
            print("Time per one file = ", (time.time() - start_time_all)/1000, ' seconds')
        print('Files preparation with proccess_zip is over\n')
        print('Full Time = ', (time.time() - start_time_all)/1000 , ' seconds')
        print("End chunk = ",chunk)
        # End inserting Nail

    def consolidate_games(self, name, samples):
        files_needed = set(file_name for file_name, index in samples)
        file_names = []
        for zip_file_name in files_needed:
            file_name = zip_file_name.replace('.tar.gz', '') + name
            file_names.append(file_name)

        feature_list = []
        label_list = []
        for file_name in file_names:
            file_prefix = file_name.replace('.tar.gz', '')
            base = self.data_dir + '/' + file_prefix + '_features_*.npy'
            for feature_file in glob.glob(base):
                label_file = feature_file.replace('features', 'labels')
                x = np.load(feature_file)
                y = np.load(label_file)
                x = x.astype('float32')
                y = to_categorical(y.astype(int), 19 * 19)
                feature_list.append(x)
                label_list.append(y)

        features = np.concatenate(feature_list, axis=0)
        labels = np.concatenate(label_list, axis=0)

        feature_file = self.data_dir + '/' + name
        label_file = self.data_dir + '/' + name

        np.save(feature_file, features)
        np.save(label_file, labels)

        return features, labels

    @staticmethod
    def get_handicap(sgf):  # Get handicap stones
        go_board = Board(19, 19)


        first_move_done = False
        move = None
        game_state = GameState.new_game(19)
        board_ext = Board_Ext(game_state.board)
        if sgf.get_handicap() is not None and sgf.get_handicap() != 0:
            for setup in sgf.get_root().get_setup_stones():
                for move in setup:
                    row, col = move
                    go_board.place_stone(Player.black, Point(row + 1, col + 1))  # black gets handicap
                    #My inserting Nail
                    point = Point(row + 1, col + 1)
                    ret = board_ext.place_stone_ext( go_board, 'b', point)  # Handicap for black Player
                    #### Nail

            first_move_done = True
            game_state = GameState(go_board, Player.white, None, move)
        return game_state, first_move_done, board_ext

    def map_to_workers(self, data_type, samples):
        zip_names = set()
        indices_by_zip_name = {}
        for filename, index in samples:
            zip_names.add(filename)
            if filename not in indices_by_zip_name:
                indices_by_zip_name[filename] = []
            indices_by_zip_name[filename].append(index)

        zips_to_process = []
        for zip_name in zip_names:
            base_name = zip_name.replace('.tar.gz', '')
            data_file_name = base_name + data_type
            if not os.path.isfile(self.data_dir + '/' + data_file_name):
                zips_to_process.append((self.__class__, self.encoder_string, zip_name,
                                        data_file_name, indices_by_zip_name[zip_name]))

        cores = multiprocessing.cpu_count()  # Determine number of CPU cores and split work load among them
        pool = multiprocessing.Pool(processes=cores)
        p = pool.map_async(worker, zips_to_process)
        try:
            _ = p.get()
        except KeyboardInterrupt:  # Caught keyboard interrupt, terminating workers
            pool.terminate()
            pool.join()
            sys.exit(-1)

    def num_total_examples(self, zip_file, game_list, name_list):
        total_examples = 0
        for index in game_list:
            name = name_list[index + 1]
            if name.endswith('.sgf'):
                sgf_content = zip_file.extractfile(name).read()
                sgf = Sgf_game.from_string(sgf_content)
                game_state, first_move_done, board_ext = self.get_handicap(sgf)

                num_moves = 0
                for item in sgf.main_sequence_iter():
                    color, move = item.get_move()
                    if color is not None:
                        if first_move_done:
                            num_moves += 1
                        first_move_done = True
                total_examples = total_examples + num_moves
            else:
                raise ValueError(name + ' is not a valid sgf')
        return total_examples

if __name__ == '__main__':
    go_board_rows, go_board_cols = 19, 19
    num_classes = go_board_rows * go_board_cols

    encoder = MySevenPlaneEncoder_S((go_board_rows, go_board_cols))
    num_games = 100000
    start_time = time.time()
    processor = GoDataProcessor(encoder=encoder.name(), data_directory='data', count_stones_debute=None,
                                count_stones_middle=None, count_stones_end=None)
    generator = processor.load_go_data('train', num_games, use_generator=True)
    test_generator = processor.load_go_data('test', num_games, use_generator=True)

    end_time= time.time()
    print('Prepare data finished, time = ', (end_time-start_time)/1000, ' seconds')

