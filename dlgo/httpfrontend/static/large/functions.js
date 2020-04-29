function resetGame(ev) {
    jrecord.jboard.clear();
    jrecord.root = jrecord.current = null;
    jrecord.info = {};
    record = [];
    waitingForBot = false;
    ev.preventDefault();
}
function coordsToString(point) {
    var row = (BOARD_SIZE - 1) - point.j;
    var col = point.i;
    return colnames[col] + ((row + 1).toString());
}
function stringToCoords(move_string) {
    var colStr = move_string.substring(0, 1);
    var rowStr = move_string.substring(1);
    var col = colnames.indexOf(colStr);
    var row = BOARD_SIZE - parseInt(rowStr, 10);
    return new JGO.Coordinate(col, row);
}
function applyMove(player, coord) {
    var play = jboard.playMove(coord, player, ko);
    if (play.success) {
        record.push(coordsToString(coord));
        node = jrecord.createNode(true);
        node.info.captures[player] += play.captures.length; // tally captures
        node.setType(coord, player); // play stone
        node.setType(play.captures, JGO.CLEAR); // clear opponent's stones
        if (lastMove) {
            node.setMark(lastMove, JGO.MARK.NONE); // clear previous mark
        }
        if (ko) {
            node.setMark(ko, JGO.MARK.NONE); // clear previous ko mark
        }
        node.setMark(coord, JGO.MARK.CIRCLE); // mark move
        lastMove = coord;
      if(play.ko)
        node.setMark(play.ko, JGO.MARK.CIRCLE); // mark ko, too
      ko = play.ko;
    } else alert('Illegal move: ' + play.errorMsg);
}
function waitForBot() {
    console.log('Waiting for bot...');
    document.getElementById('status').style.display = 'none';
    document.getElementById('spinner').style.display = 'block';
    waitingForBot = true;
}
function stopWaiting(botmove) {
    var text = 'Bot plays ' + botmove;
    if (botmove == 'pass') {
        text = 'Bot passes';
    } else if (botmove == 'resign') {
        text = 'Bot resigns';
    }
    document.getElementById('status').innerHTML = text;
    document.getElementById('status').style.display = 'block';
    document.getElementById('spinner').style.display = 'none';
    waitingForBot = false;
}