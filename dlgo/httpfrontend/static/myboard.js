var BOARD_SIZE = 19;
var jrecord = new JGO.Record(BOARD_SIZE);
var jboard = jrecord.jboard;
var jsetup = new JGO.Setup(jboard, JGO.BOARD.largeWalnut);
var player = JGO.BLACK; // next player
var ko = false, lastMove = false; // ko coordinate and last move coordinate
var lastHover = false, lastX = -1, lastY = -1; // hover helper vars
var record = [];
var colnames = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T'];
var waitingForBot = false;

jsetup.setOptions({ stars: { points: 5 } });
jsetup.create('board', function (canvas) {
    canvas.addListener('click', function (coord, ev) {
        if (waitingForBot) {
            return;
        }
        var opponent = (player == JGO.BLACK) ? JGO.WHITE : JGO.BLACK;
        if (ev.shiftKey) { // on shift do edit
            if (jboard.getMark(coord) == JGO.MARK.NONE)
                jboard.setMark(coord, JGO.MARK.SELECTED);
            else
                jboard.setMark(coord, JGO.MARK.NONE);
            return;
        }
        // clear hover away - it'll be replaced or then it will be an illegal move
        // in any case so no need to worry about putting it back afterwards
        if (lastHover)
            jboard.setType(new JGO.Coordinate(lastX, lastY), JGO.CLEAR);
        lastHover = false;
        console.log('Human', coordsToString(coord));
        applyMove(JGO.BLACK, coord);
        waitForBot();
        fetch('/select-move/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ 'board_size': BOARD_SIZE, 'moves': record }),
        }).then(function (response) {
            if (!waitingForBot) {
                console.log('Got response but not waiting for one');
                return;
            }
            response.json().then(function (data) {
                //  Score
                console.log('score = ', data.score, 'data: ', data)
                let score_go = document.getElementById("score")
               
                score_go.style.fontWeight="bolder"
                score_go.style.color="red"
                score_go.innerHTML = 'Выигрывают ' + data.winner + ', счет: ' + data.score;
                let territory_black = document.getElementById("black_score")
                territory_black.innerHTML = "Территория черных: " + data.territory_black
                let territory_white = document.getElementById("white_score")
                territory_white.innerHTML = "Территория белых: " + data.territory_white

                if (data.bot_move == 'pass' || data.bot_move == 'resign') {
                    record.push(data.bot_move);
                   
                } else {
                    var botCoord = stringToCoords(data.bot_move);
                    applyMove(JGO.WHITE, botCoord);
                    
                }
                stopWaiting(data.bot_move); // Здесь отображается ход бота
            });
        }).catch(function (error) {
            console.log(error);
            stopWaiting(data.bot_move);
        });
    });
    canvas.addListener('mousemove', function (coord, ev) {
        if (coord.i == -1 || coord.j == -1 || (coord.i == lastX && coord.j == lastY))
            return;
        if (lastHover) // clear previous hover if there was one
            jboard.setType(new JGO.Coordinate(lastX, lastY), JGO.CLEAR);
        lastX = coord.i;
        lastY = coord.j;
        if (jboard.getType(coord) == JGO.CLEAR && jboard.getMark(coord) == JGO.MARK.NONE) {
            jboard.setType(coord, player == JGO.WHITE ? JGO.DIM_WHITE : JGO.DIM_BLACK);
            lastHover = true;
        } else
            lastHover = false;
    });
    canvas.addListener('mouseout', function (ev) {
        if (lastHover)
            jboard.setType(new JGO.Coordinate(lastX, lastY), JGO.CLEAR);
        lastHover = false;
    });
});