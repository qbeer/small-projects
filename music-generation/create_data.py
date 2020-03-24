from music21 import corpus

paths = corpus.getComposer('bach')

music = [corpus.parse(path) for path in paths[:1]]

chord = music[0].chordify(addTies=False)

fp = chord.write('midi', fp='test.mid')