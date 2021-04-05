#
# A GUI for ps-auxw's CLI-P (commandline driven semantic image search using OpenAI's CLIP)
#

import sys
import time
from io import StringIO
import contextlib
from pathlib import Path
import collections

from PyQt5.QtCore import (
    pyqtSignal, pyqtSlot,
    Qt,
    QMetaObject, QObject, QThread,
    QSize,
    QItemSelectionModel, QIdentityProxyModel,
    QTimer,
)
from PyQt5.QtGui import (
    QStandardItemModel, QStandardItem,
    QImage, QPixmap,
)
from PyQt5.QtWidgets import (
    qApp,
    QApplication, QMainWindow, QWidget,
    QSizePolicy,
    QHBoxLayout, QVBoxLayout, QTabWidget, QToolBar,
    QComboBox, QLabel, QPushButton, QTextEdit,
    QListView, QTableView,
)

# Load delayed, so the GUI is already visible,
# as this may take a long time.
query_index = None

def imageFromOpenCV(image):
    """Converts an OpenCV (cv2) image (as generated by query-index.Search.prepare_image())
    to Qt. That is, prepares image for display in the GUI.
    """
    if image is None:
        raise TypeError("OpenCV image argument required")
    return QImage(image.data, image.shape[1], image.shape[0], 3 * image.shape[1], QImage.Format_RGB888).rgbSwapped()

class ImagesWorker(QObject):
    imageLoaded = pyqtSignal(int, str, QImage)
    largeThumbnailReady = pyqtSignal(int, str, QImage)
    smallThumbnailReady = pyqtSignal(int, str, QImage)
    logMessage = pyqtSignal(str)

    def __init__(self, largeSize=180, smallSize=24, parent=None):
        super().__init__(parent)
        self._largeSize = largeSize
        self._smallSize = smallSize
        self._queue = collections.deque()
        self._timerWrapped = None

    def largeSize(self):
        return self._largeSize
    def setLargeSize(self, size):
        self._largeSize = size
    def smallSize(self):
        return self._smallSize
    def setSmallSize(self, size):
        self._smallSize = size

    def _timer(self):
        if self._timerWrapped is None:
            # Zero-second timer, after starting it will run continously after any events have been processed.
            self._timerWrapped = QTimer(self)
            self._timerWrapped.timeout.connect(self._doLoadImage)
        return self._timerWrapped

    @pyqtSlot(int, str)
    def loadImage(self, fix_idx, tfn):
        # Check if already queued.
        for other_fix_idx, other_tfn in self._queue:
            if other_fix_idx == fix_idx and other_tfn == tfn:
                return
        # Otherwise, queue now & process later.
        self._queue.appendleft((fix_idx, tfn))
        timer = self._timer()
        if not timer.isActive():
            timer.start()

    @pyqtSlot()
    def clearQueue(self):
        self._queue.clear()
        timer = self._timer()
        if timer.isActive():
            timer.stop()

    @pyqtSlot()
    def _doLoadImage(self):
        if len(self._queue) == 0:
            self._timer().stop()
            return
        fix_idx, tfn = self._queue.pop()
        imageCv2 = None
        try:
            imageCv2, _ = query_index.Search.load_image(tfn=tfn, max_res=None)
        except Exception as ex:
            self.logMessage.emit(f"Got exception {type(ex).__name__} loading image: {ex}")
        if imageCv2 is None:
            return
        imageQt = None
        try:
            imageQt = imageFromOpenCV(imageCv2)
        except Exception as ex:
            self.logMessage.emit(f"Got exception {type(ex).__name__} converting image: {ex}")
        if imageQt is None:
            return
        self.imageLoaded.emit(fix_idx, tfn, imageQt)
        largeThumbnail = imageQt.scaled(QSize(self._largeSize, self._largeSize), Qt.KeepAspectRatio)
        self.largeThumbnailReady.emit(fix_idx, tfn, largeThumbnail)
        smallThumbnail = imageQt.scaled(QSize(self._smallSize, self._smallSize), Qt.KeepAspectRatio)
        self.smallThumbnailReady.emit(fix_idx, tfn, smallThumbnail)

class HistoryComboBox(QComboBox):
    pageChangeRequested = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(HistoryComboBox, self).__init__(parent)
        self.isShowingPopup = False
        self._defaultButton = None

    def defaultButton(self):
        return self._defaultButton

    def setDefaultButton(self, button):
        self._defaultButton = button

    def showPopup(self):
        super(HistoryComboBox, self).showPopup()
        self.isShowingPopup = True

    def hidePopup(self):
        self.isShowingPopup = False
        super(HistoryComboBox, self).hidePopup()

    def keyPressEvent(self, ev):
        key = ev.key()
        # On Return (Here, Enter is located on the key pad, instead!),
        # activate the associated default button.
        # Necessary in non-dialogs.
        if key == Qt.Key_Return and self._defaultButton != None:
            # Propagate further, first.
            # Necessary so the user input still gets added to the list.
            super(HistoryComboBox, self).keyPressEvent(ev)
            # Then, activate the default button.
            self._defaultButton.click()
        # On up/down, ensure the popup opens.
        elif (key == Qt.Key_Up or key == Qt.Key_Down) and not self.isShowingPopup:
            self.showPopup()
            # Don't prevent default handling of the key press.
            super(HistoryComboBox, self).keyPressEvent(ev)
        # On PageUp/Down, emit a signal
        # (so this can control, e.g., scrolling of the console log).
        elif key == Qt.Key_PageUp or key == Qt.Key_PageDown:
            self.pageChangeRequested.emit(key == Qt.Key_PageUp)
            # Don't prevent default handling of the key press.
            super(HistoryComboBox, self).keyPressEvent(ev)
        # Otherwise, propagate key press further.
        else:
            super(HistoryComboBox, self).keyPressEvent(ev)

class MainWindow(QMainWindow):
    class OurTabPage(QWidget):
        resized = pyqtSignal()
        def __init__(self, parent=None):
            super(MainWindow.OurTabPage, self).__init__(parent)
        def resizeEvent(self, ev):
            super(MainWindow.OurTabPage, self).resizeEvent(ev)
            self.resized.emit()

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.imagesWorker = None
        self.imagesWorkerThread = None
        self.search = None
        self.searchResultSelected = None

        # TODO: Take from config db.
        self.resize(1600, 900)

        centralWidget = QWidget()
        centralVBox = QVBoxLayout(centralWidget)

        self.tabWidget = QTabWidget()


        # Page 1: Console
        self.consoleTabPage = QWidget()
        consoleVBox = QVBoxLayout(self.consoleTabPage)

        self.infoLabel = QLabel(
            "ps-auxw says, \"CLI-P is commandline driven semantic image search using OpenAI's CLIP\"\n"
            "canvon says, \"This is a GUI for ps-auxw's CLI-P\"")
        self.searchOutput = QTextEdit()
        self.searchOutput.setReadOnly(True)

        consoleVBox.addWidget(self.infoLabel)
        consoleVBox.addWidget(self.searchOutput)
        self.tabWidget.addTab(self.consoleTabPage, "&1 Console")


        # Page 2: Images
        self.imagesTabPage = self.OurTabPage()
        self.imagesTabPage.resized.connect(self.imagesTabPageResized)
        imagesVBox = QVBoxLayout(self.imagesTabPage)

        imgsToolBar = QToolBar()
        self.imagesToolBar = imgsToolBar
        self.imagesActionAddTag = imgsToolBar.addAction("Add to tag (&+)", self.imagesActionAddTagTriggered)
        self.imagesActionDelTag = imgsToolBar.addAction("Del from tag (&-)", self.imagesActionDelTagTriggered)
        self.imagesActionAddTag.setShortcut("Ctrl+T")
        self.imagesActionDelTag.setShortcut("Ctrl+Shift+T")

        self.imageLabel = QLabel()
        self.imagesTableView = QTableView()
        self.imagesTableView.setEditTriggers(QTableView.NoEditTriggers)
        self.imagesTableView.activated.connect(self.searchResultsActivated)

        imagesVBox.addWidget(self.imagesToolBar)
        imagesVBox.addWidget(self.imageLabel)
        imagesVBox.addWidget(self.imagesTableView)
        self.tabWidget.addTab(self.imagesTabPage, "&2 Images")


        # Page 3: Thumbnails
        self.thumbnailsTabPage = QWidget()
        thumbnailsVBox = QVBoxLayout(self.thumbnailsTabPage)

        self.thumbnailsListView = QListView()
        self.thumbnailsListView.setEditTriggers(QListView.NoEditTriggers)
        self.thumbnailsListView.setViewMode(QListView.IconMode)
        self.thumbnailsListView.activated.connect(self.thumbnailsActivated)

        thumbnailsVBox.addWidget(self.thumbnailsListView)
        self.tabWidget.addTab(self.thumbnailsTabPage, "&3 Thumbnails")


        self.searchHint = QLabel()


        # Search input box & go button
        inputHBox = QHBoxLayout()

        self.searchInputLabel = QLabel("&Search")
        self.searchInputLabel.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.searchInput = HistoryComboBox()
        self.searchInput.setEditable(True)
        self.searchInput.pageChangeRequested.connect(self.searchInputPageChangeRequested)
        #
        # This fired too often, but we only want to search
        # when the user finally hits return...
        #self.searchInput.activated.connect(self.handleSearchInput)
        #
        self.searchInputLabel.setBuddy(self.searchInput)

        self.searchInputButton = QPushButton()
        #
        # Doesn't work without a Dialog:
        #self.searchInputButton.setAutoDefault(True)
        #self.searchInputButton.setDefault(True)
        # ..so, do this instead:
        self.searchInput.setDefaultButton(self.searchInputButton)
        #
        self.searchInputButton.setText("&Go")
        self.searchInputButton.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.searchInputButton.clicked.connect(self.handleSearchInput)

        inputHBox.addWidget(self.searchInputLabel)
        inputHBox.addWidget(self.searchInput)
        inputHBox.addWidget(self.searchInputButton)


        centralVBox.addWidget(self.tabWidget)
        centralVBox.addWidget(self.searchHint)
        centralVBox.addLayout(inputHBox)

        self.setCentralWidget(centralWidget)
        self.searchInput.setFocus()

        self.createSearchResultsModel()

    def imagesTabPageResized(self):
        contents = self.imagesTabPage.contentsRect()
        self.imageLabel.setMaximumSize(contents.width(), contents.height() * 8 / 10)  # 80%
        self.imageLabel.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Maximum)

    def searchInputPageChangeRequested(self, pageUp):
        w = self.tabWidget.currentWidget()
        if w is self.consoleTabPage:
            # Scroll console log.
            out = self.searchOutput
            height = out.viewport().height()
            # This didn't work:
            #out.scrollContentsBy(0, height * (-1 if pageUp else 1))
            # This worked, but doesn't provide overlap:
            #out.verticalScrollBar().triggerAction(QScrollBar.SliderPageStepSub if pageUp else QScrollBar.SliderPageStepAdd)
            # So we end up with this:
            # (It scrolls by half the viewport height.)
            vbar = out.verticalScrollBar()
            vbar.setValue(vbar.value() + height * (-1 if pageUp else 1) / 2)
        elif w is self.imagesTabPage:
            # Scroll in & activate search results.
            view = self.imagesTableView
            model = view.model()
            selectionModel = view.selectionModel()
            index = selectionModel.currentIndex()
            nextIndex = None
            if not index.isValid():
                nextIndex = model.index(0, 0)
            else:
                nextIndex = index.siblingAtRow(index.row() + (-1 if pageUp else 1))
            if not nextIndex.isValid():
                return
            selectionModel.setCurrentIndex(nextIndex, QItemSelectionModel.SelectCurrent)
            self.searchResultsActivated(nextIndex)

    def loadModules(self):
        global query_index
        if query_index is None:
            self.appendSearchOutput("Loading query-index...")
            qApp.processEvents()
            loadStart = time.perf_counter()
            query_index = __import__('query-index')  # TODO: Adjust file name.
            loadTime = time.perf_counter() - loadStart
            self.appendSearchOutput(f"Loaded query-index: {loadTime:.4f}s")
            qApp.processEvents()
        if self.search is None:
            self.appendSearchOutput("Instantiating search...")
            qApp.processEvents()
            instantiateStart = time.perf_counter()
            self.search = query_index.Search()
            instantiateTime = time.perf_counter() - instantiateStart
            self.appendSearchOutput(f"Instantiated search: {instantiateTime:.4f}s")
            qApp.processEvents()

            self.appendSearchOutput("\n" + self.search.init_msg)
            self.searchHint.setText("Short help: " + self.search.prompt_prefix)

    def showEvent(self, ev):
        super(MainWindow, self).showEvent(ev)
        QTimer.singleShot(0, self.delayLoadModules)  # (Run after all events.)

    def delayLoadModules(self):
        QTimer.singleShot(50, self.loadModules)  # (Delay a bit further in the hopes it might actually work.)

    def appendSearchOutput(self, lines):
        # Skip calls with nothing to convey.
        if lines is None or lines == "":
            return
        # Strip last newline, but only that.
        # That way, an empty line at the end
        # can be requested by "...\n\n".
        # Otherwise, we could simply use: lines.rstrip('\n')
        if lines[-1] == '\n':
            lines = lines[:-1]
        self.searchOutput.append(lines)

    def stdoutSearchOutput(self, code):
        ret = None
        with contextlib.closing(StringIO()) as f:
            with contextlib.redirect_stdout(f):
                ret = code()
            self.appendSearchOutput(f.getvalue())
        return ret

    def handleSearchInput(self):
        inputText  = self.searchInput.currentText()
        inputIndex = self.searchInput.currentIndex()
        storedText = None if inputIndex == -1 else self.searchInput.itemText(inputIndex)

        search = self.search
        if search is None:
            self.appendSearchOutput("Search not ready, yet...")
            return
        self.appendSearchOutput(">>> " + inputText)
        search.in_text = inputText.strip()
        if storedText != inputText:
            self.searchInput.addItem(inputText)
        self.searchInput.clearEditText()
        self.clearSearchResultsModel()

        iteration_done = self.stdoutSearchOutput(search.do_command)
        if iteration_done:
            # Check for q (quit) command.
            if search.running_cli == False:
                self.close()
            return

        self.stdoutSearchOutput(search.do_search)
        n_results = 0 if search.results is None else len(search.results)
        if not n_results > 0:
            self.appendSearchOutput("No results.")
            return
        #self.stdoutSearchOutput(search.do_display)
        self.appendSearchOutput(f"Building results model for {n_results} results...")
        for result in search.prepare_results():
            self.appendToSearchResultsModel(result)
        self.appendSearchOutput(f"Built results model with {self.searchResultsModel.rowCount()} entries.")

        # Already activate Images tab and load first image...
        self.tabWidget.setCurrentWidget(self.imagesTabPage)
        view = self.imagesTableView
        nextIndex = view.model().index(0, 0)
        if nextIndex.isValid():
            view.selectionModel().setCurrentIndex(nextIndex, QItemSelectionModel.SelectCurrent)
            self.searchResultsActivated(nextIndex)

    class SearchResultsModel(QStandardItemModel):
        def __init__(self, rows, columns, *, mainWindow, parent=None):
            super().__init__(rows, columns, parent)
            self._mainWindow = mainWindow
        def data(self, index, role):
            superData = super().data(index, role)
            if not (
                index is not None and
                index.isValid() and
                index.column() == 1 and
                role == Qt.DecorationRole):
                # Only act ourselves on request for icon. Pass through everything else.
                return superData
            if superData is not None:
                # Use existing icon.
                return superData
            # Request load thumbnail.
            result = super().data(index, Qt.UserRole + 1)
            if result is None:
                return None
            self._mainWindow.imageToLoad.emit(result.fix_idx, result.tfn)
            return None

    class ThumbnailsProxyModel(QIdentityProxyModel):
        def __init__(self, mainWindow=None, parent=None):
            super().__init__(parent=parent)
            self._mainWindow = mainWindow
            self._thumbnails = {}
            self._tempPixmap = None
        def mainWindow(self):
            return self._mainWindow
        def tempPixmap(self):
            return self._tempPixmap
        def setTempPixmap(self, pixmap):
            self._tempPixmap = pixmap
        def clearCache(self):
            self._thumbnails.clear()
        def thumbnail(self, row, result=None):
            """Get thumbnail; if not already cached, use parameter result as info for placing a request for asynchronous loading."""
            if row not in self._thumbnails:
                self._mainWindow.imageToLoad.emit(result.fix_idx, result.tfn)
                return self._tempPixmap
            return self._thumbnails[row]
        def setThumbnail(self, row, thumbnail):
            """Set (override potentially already cached) thumbnail."""
            self._thumbnails[row] = thumbnail
        def data(self, index, role):
            """Overridden proxy model data() which overlays column 1 (fix_idx + thumbnail) data."""
            if index.parent().isValid() or index.column() != 1:  # fix_idx + thumbnail
                return super().data(index, role)
            row = index.row()
            result = super().data(index, Qt.UserRole + 1)
            if role == Qt.UserRole + 1:
                return result
            elif role == Qt.DisplayRole:
                # Overlay single-string rendering of result.
                return "(result missing)" if result is None else f"{result.fix_idx}: {Path(result.tfn).name}"
            elif role == Qt.DecorationRole:
                # Overlay a larger thumbnail/icon, which potentially has already been cached...
                return None if result is None else self.thumbnail(row, result)
            else:
                # We could try to pass more things through, but let's rather
                # define this proxy model's column's meaning fully ourselves!
                return None
        def setData(self, index, data, role):
            if not (
                index is not None and
                index.isValid() and
                index.column() == 1 and
                role is not None and
                role == Qt.DecorationRole):
                # Reject update.
                indexInfo = None
                if index.isValid():
                    indexInfo = {
                        'row': index.row(),
                        'column': index.column(),
                        'hasParent': index.parent().isValid(),
                    }
                raise RuntimeError(f"Invalid use of {type(self).__name__}.setData(), with indexInfo={indexInfo!r}, data={data!r}, role={role!r}")
            self.setThumbnail(index.row(), data)
            self.dataChanged.emit(index, index, [role])

    imageToLoad = pyqtSignal(int, str)

    def createSearchResultsModel(self):
        model = self.SearchResultsModel(0, 4, mainWindow=self)
        model.setHorizontalHeaderLabels(["score", "fix_idx", "face_id", "filename"])
        self.searchResultsModel = model
        self.imagesTableView.setModel(model)
        self.imagesTableView.horizontalHeader().setStretchLastSection(True)
        thumbsModel = self.ThumbnailsProxyModel(mainWindow=self)
        thumbsModel.setSourceModel(model)
        self.searchResultsThumbnailsProxyModel = thumbsModel
        self.thumbnailsListView.setModel(thumbsModel)
        self.thumbnailsListView.setModelColumn(1)  # fix_idx + thumbnail

        # Prepare for asynchronous thumbnail loading...
        self.imagesWorkerThread = QThread()
        self.imagesWorkerThread.setObjectName("ImagesWorker")
        self.imagesWorker = ImagesWorker()
        #
        # Try to avoid the situation that "all" results fit on the screen
        # on Thumbnails tab at once while they have no icon, yet,
        # so are just a tiny piece of text...
        largeSize = self.imagesWorker.largeSize()
        largeSquare = QSize(largeSize, largeSize)
        #self.thumbnailsListView.setIconSize(largeSquare)  # This doesn't seem to help.
        tempImage = QImage(largeSquare, QImage.Format_RGB888)
        tempImage.fill(Qt.lightGray)
        thumbsModel.setTempPixmap(QPixmap.fromImage(tempImage))
        #
        self.imagesWorker.moveToThread(self.imagesWorkerThread)
        self.imageToLoad.connect(self.imagesWorker.loadImage)
        self.imagesWorker.logMessage.connect(self.appendSearchOutput)
        self.imagesWorker.largeThumbnailReady.connect(self.handleLargeThumbnail)
        self.imagesWorker.smallThumbnailReady.connect(self.handleSmallThumbnail)
        self.imagesWorkerThread.start()

    def clearSearchResultsModel(self):
        self.searchResultSelected = None
        # Avoid uselessly generating thumbnails that will never be seen.
        QMetaObject.invokeMethod(self.imagesWorker, 'clearQueue', Qt.BlockingQueuedConnection)
        # Avoid stale thumbnails in future search results.
        self.searchResultsThumbnailsProxyModel.clearCache()
        #
        # Clear the search results model itself.
        #
        # Don't use clear(), as that will get rid of the header labels
        # and column count, too...
        #self.searchResultsModel.clear()
        self.searchResultsModel.setRowCount(0)
        #
        # Stop displaying a previous search'es result image.
        self.imageLabel.clear()

    def prepareSearchResultsModelEntry(self, result):
        scoreItem  = QStandardItem(str(result.score))
        fixIdxItem = QStandardItem(str(result.fix_idx))
        faceIdItem = QStandardItem(str(result.face_id))
        tfnItem    = QStandardItem(str(result.tfn))
        items = [scoreItem, fixIdxItem, faceIdItem, tfnItem]
        for item in items:
            item.setData(result)
        return items

    def handleLargeThumbnail(self, fix_idx, tfn, imageQt):
        update = self._handleThumbnail(self.searchResultsThumbnailsProxyModel, fix_idx, tfn, imageQt)
        if update:
            # Reflow... Needed to avoid rendering bugs.
            self.thumbnailsListView.setFlow(self.thumbnailsListView.flow())

    def handleSmallThumbnail(self, fix_idx, tfn, imageQt):
        self._handleThumbnail(self.searchResultsModel, fix_idx, tfn, imageQt)

    def _handleThumbnail(self, model, fix_idx, tfn, imageQt):
        update = False
        for row in range(model.rowCount()):
            column = 1
            modelIndex = model.index(row, column)
            if not modelIndex.isValid():
                continue
            result = modelIndex.data(Qt.UserRole + 1)
            if result is None:
                continue
            if result.fix_idx != fix_idx:
                continue
            # (Check the filename, too?)
            #
            update = True
            model.setData(modelIndex, QPixmap.fromImage(imageQt), Qt.DecorationRole)
            # Continue search in case we have duplicate results (e.g., from the `l` command).
        return update

    def appendToSearchResultsModel(self, result):
        model = self.searchResultsModel
        items = self.prepareSearchResultsModelEntry(result)
        result.gui_rowOffset = model.rowCount()
        model.appendRow(items)

    def recreateSearchResultsModelRow(self, result):
        search = self.search
        if search is None:
            self.appendSearchOutput("Search instance missing.")
            return
        rowOffset = result.gui_rowOffset
        # Recreate Search.Result instance.
        # (e.g., rereads annotations/tags.)
        search.tried_j = -1
        search.last_vector = None
        recreatedResult, j, _ = search.prepare_result(result.results_j)
        if j is None:
            self.appendSearchOutput(f"Failed to recreate search results model row {rowOffset+1}: Prepare result indicated end of results.")
            return
        elif recreatedResult is None:
            self.appendSearchOutput(f"Failed to recreate search results model row {rowOffset+1}: Prepare result indicated skip.")
            return
        recreatedResult.gui_rowOffset = rowOffset
        # Update Qt-side model.
        model = self.searchResultsModel
        items = self.prepareSearchResultsModelEntry(recreatedResult)
        for columnOffset in range(model.columnCount()):
            model.setItem(rowOffset, columnOffset, items[columnOffset])
        return recreatedResult

    def searchResultsActivated(self, index):
        result = index.data(Qt.UserRole + 1)
        self.showSearchResult(result, force=True)

    def thumbnailsActivated(self):
        curIndex = self.thumbnailsListView.selectionModel().currentIndex()
        if not curIndex.isValid():
            return
        self.tabWidget.setCurrentWidget(self.imagesTabPage)
        self.imagesTableView.selectionModel().setCurrentIndex(curIndex, QItemSelectionModel.SelectCurrent)
        self.searchResultsActivated(curIndex)

    def showSearchResult(self, result, force=False):
        if not force and self.searchResultSelected is result:
            return
        self.searchResultSelected = result
        if result is None:
            return
        self.appendSearchOutput(result.format_output())
        # Prepare image.
        try:
            size = self.imageLabel.maximumSize()
            max_res = (size.width(), size.height())
            image = self.search.prepare_image(result, max_res=max_res)
            if image is None:
                raise RuntimeError("No image.")
        except Exception as ex:
            self.appendSearchOutput(f"Error preparing image: {ex}")
            return
        # Convert prepared image to Qt/GUI.
        qtImage = imageFromOpenCV(image)
        self.imageLabel.setPixmap(QPixmap.fromImage(qtImage))

    def getThumbnail(self, result, *, size):
        try:
            image = self.search.prepare_image(result, max_res=size, show_faces=False)
            if image is None:
                raise RuntimeError("No result image.")
            qtImage = imageFromOpenCV(image)
            if qtImage is None:
                raise RuntimeError("No Qt image.")
            pixmap = QPixmap.fromImage(qtImage)
            if pixmap is None:
                raise RuntimeError("No Qt pixmap.")
            return pixmap
        except Exception as ex:
            raise RuntimeError(f"Loading thumbnail from result number {result.results_j + 1} image {result.fix_idx} file name {result.tfn!r} failed") from ex

    def updateSearchResultSelected(self, updateCode):
        result = self.searchResultSelected
        if result is None:
            self.appendSearchOutput("Update search result selected: No search result selected.")
            return
        self.stdoutSearchOutput(lambda: updateCode(result))
        recreatedResult = self.recreateSearchResultsModelRow(result)
        if recreatedResult is None:
            return
        self.showSearchResult(recreatedResult, force=True)

    def imagesActionAddTagTriggered(self):
        search = self.search
        if search is None:
            self.appendSearchOutput("Search instance missing.")
            return
        self.updateSearchResultSelected(search.maybe_add_tag)

    def imagesActionDelTagTriggered(self):
        search = self.search
        if search is None:
            self.appendSearchOutput("Search instance missing.")
            return
        self.updateSearchResultSelected(search.maybe_del_tag)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName("CLI-P GUI")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
